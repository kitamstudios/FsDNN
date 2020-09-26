namespace KS.FsDNN

open System
open System.Diagnostics

[<AutoOpen>]
module TrainerDomain =
  type EpochCallback = int -> TimeSpan -> Tensor<double> -> unit

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

  type HyperParameters =
    { Epochs : int
      LearningRate: Tensor<double>
      Lambda: Tensor<double> option
      Optimizer: Optimizer
      BatchSize: BatchSize } with
    static member Defaults =
      { Epochs = 1_000
        LearningRate = TensorR0 0.01
        Lambda = None (* Some 0.7 *)
        Optimizer = NullOptimizer (* ADAMOptimization ADAMParameters.Defaults *)
        BatchSize = BatchSizeAll (* BatchSize64 *) }

module Trainer =

  let private _getMiniBatches (batchSize: BatchSize) (X: Tensor<double>, Y: Tensor<double>): (Tensor<double> * Tensor<double>) seq =
    if batchSize = BatchSizeAll then
      seq {
        yield X, Y
      }
    else
      Prelude.undefined

  let private _trainNetworkFor1MiniBatch net lr (_: Tensor<double>, ts: TrainingState, timer: Stopwatch) (X: Tensor<double>, Y: Tensor<double>): (Tensor<double> * TrainingState * Stopwatch) =
    timer.Start()
    let J', cache = Net.forwardPropagate { net with Parameters = ts.Parameters } X Y
    let gradients = Net.backPropagate net cache
    let ts = Optimizer.updateParameters lr ts gradients
    timer.Stop()
    let m = X.ColumnCount
    let J = (m |> double |> TensorR0).PointwiseMultiply(J')
    J, ts, timer

  let private _trainNetworkFor1Epoch (timer: Stopwatch) (callback: EpochCallback) (net: Net) (X: Tensor<double>) (Y: Tensor<double>) (hp: HyperParameters) (ts: TrainingState) epoch =
    timer.Restart()

    let J, ts, timer =
      (X, Y)
      |> _getMiniBatches hp.BatchSize
      |> Seq.fold (_trainNetworkFor1MiniBatch net hp.LearningRate) (0. |> TensorR0, ts, timer)
    timer.Stop()

    let m = double X.ColumnCount
    let J = (TensorR0 (1. / m)).PointwiseMultiply(J)
    callback epoch timer.Elapsed J
    ts

  let trainWithGD (callback: EpochCallback) (net: Net) (X: Tensor<double>) (Y: Tensor<double>) (hp: HyperParameters) : Net =
    if X.ColumnCount <> Y.ColumnCount then
      failwithf "Column count of X (%d) is not the same as that of Y (%d)." X.ColumnCount Y.ColumnCount

    let timer = Stopwatch()

    let ts0 = Optimizer.initializeState hp.Optimizer net.Parameters

    let ts =
      seq { for epoch in 0 .. (hp.Epochs - 1) do epoch }
      |> Seq.fold (_trainNetworkFor1Epoch timer callback net X Y hp) ts0

    { net with Parameters = ts.Parameters }
