namespace KS.FsDNN

open System
open System.Diagnostics

[<AutoOpen>]
module TrainerDomain =
  type EpochCallback = int -> TimeSpan -> Tensor<double> -> unit

  type Gradients = Map<string, Tensor<double>>

  type BatchSize =
    | BatchSize1
    | BatchSize64
    | BatchSize128
    | BatchSize256
    | BatchSize512
    | BatchSize1024
    | BatchSizeAll
    with
      member this.toInt max =
        match this with
        | BatchSize1 -> 1
        | BatchSize64 -> 64
        | BatchSize128 -> 128
        | BatchSize256 -> 256
        | BatchSize512 -> 512
        | BatchSize1024 -> 1024
        | BatchSizeAll -> max

  type MomentumParameters =
    { Beta: double }
    static member Defaults: MomentumParameters =
      { Beta = 0.9 }

  type ADAMParameters =
    { Beta1: double; Beta2: double; Epsilon: double }
    static member Defaults: ADAMParameters =
      { Beta1 = 0.9; Beta2 = 0.999; Epsilon = 1e-8 }

  type Optimization =
    | NoOptimization
    | MomentumOptimization of MomentumParameters
    | ADAMOptimization of ADAMParameters

  /// Exponentially weighted average of the gradients.
  type GradientVelocity = { dWv: Tensor<double>; dbv: Tensor<double> }
  type GradientVelocities = Map<int, GradientVelocity>

  /// Exponentially weighted average of the squared gradient.
  type SquaredGradientVelocity = { dWs: Tensor<double>; dbs: Tensor<double> }
  type SquaredGradientVelocities = Map<int, SquaredGradientVelocity>

  type TrainingState =
    | NoOptTrainingState of Parameters
    | MomentumTrainingState of Parameters * GradientVelocities
    | ADAMTrainingState of Parameters * GradientVelocities * SquaredGradientVelocities * double
    with
    member this.Parameters =
      match this with
      | NoOptTrainingState p -> p
      | MomentumTrainingState (p, _) -> p
      | ADAMTrainingState (p, _, _, _) -> p

  type HyperParameters =
    { Epochs : int
      LearningRate: Tensor<double>
      Lambda: Tensor<double> option
      Optimization: Optimization
      BatchSize: BatchSize } with
    static member Defaults =
      { Epochs = 1_000
        LearningRate = R0 0.01
        Lambda = None (* Some 0.7 *)
        Optimization = NoOptimization (* ADAMOptimization ADAMParameters.Defaults *)
        BatchSize = BatchSizeAll (* BatchSize64 *) }

module Trainer =

  let private _getMiniBatches (batchSize: BatchSize) (X: Tensor<double>, Y: Tensor<double>): (Tensor<double> * Tensor<double>) seq =
    if batchSize = BatchSizeAll then
      seq {
        yield X, Y
      }
    else
      Prelude.undefined

  let private _updateParametersWithNoOptimization (lr: Tensor<double>) (parameters: Parameters) (gradients: Gradients): TrainingState =
    parameters
    |> Map.map (fun id value -> value - (lr * gradients.[id]))
    |> NoOptTrainingState

  let private _updateParameters (hp: HyperParameters) (ts: TrainingState) (gradients: Gradients): TrainingState =
    match hp.Optimization with
    | NoOptimization -> _updateParametersWithNoOptimization hp.LearningRate ts.Parameters gradients
    | _ -> Prelude.undefined

  let private _trainNetworkFor1MiniBatch net hp (_: Tensor<double>, ts: TrainingState, timer: Stopwatch) (X: Tensor<double>, Y: Tensor<double>): (Tensor<double> * TrainingState * Stopwatch) =
    timer.Start()
    let J', cache = Net.forwardPropagate { net with Parameters = ts.Parameters } X Y
    let gradients = Net.backPropagate net cache
    let ts = _updateParameters hp ts gradients
    timer.Stop()
    let m = X.ColumnCount
    let J = (m |> double |> R0).PointwiseMultiply(J')
    J, ts, timer

  let private _trainNetworkFor1Epoch (timer: Stopwatch) (callback: EpochCallback) (net: Net) (X: Tensor<double>) (Y: Tensor<double>) (hp: HyperParameters) (ts: TrainingState) epoch =
    timer.Restart()

    let J, ts, timer =
      (X, Y)
      |> _getMiniBatches hp.BatchSize
      |> Seq.fold (_trainNetworkFor1MiniBatch net hp) (0. |> R0, ts, timer)
    timer.Stop()

    let m = double X.ColumnCount
    let J = (R0 (1. / m)).PointwiseMultiply(J)
    callback epoch timer.Elapsed J
    ts

  let trainWithGD (callback: EpochCallback) (net: Net) (X: Tensor<double>) (Y: Tensor<double>) (hp: HyperParameters) : Net =
    if X.ColumnCount <> Y.ColumnCount then
      failwithf "Column count of X (%d) is not the same as that of Y (%d)." X.ColumnCount Y.ColumnCount

    let timer = Stopwatch()

    let ts0 =
      match hp.Optimization with
      | NoOptimization -> NoOptTrainingState net.Parameters
      | _ -> Prelude.undefined

    let ts =
      seq { for epoch in 0 .. (hp.Epochs - 1) do epoch }
      |> Seq.fold (_trainNetworkFor1Epoch timer callback net X Y hp) ts0

    { net with Parameters = ts.Parameters }
