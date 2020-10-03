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
      Regularizer: Regularizer
      Optimizer: Optimizer
      BatchSize: BatchSize } with
    static member Defaults =
      { Epochs = 1_000
        LearningRate = TensorR0 0.01
        Regularizer = NullRegularizer (* L2Regularizer 0.7 *)
        Optimizer = AdaMOptimizer AdaMOptimizerDomain.AdaMParameters.Defaults
        BatchSize = BatchSizeAll (* BatchSize64 *) }

module Trainer =

  let private _getMiniBatches (batchSize: BatchSize) (X: Tensor<double>, Y: Tensor<double>): (Tensor<double> * Tensor<double>) seq =
    if batchSize = BatchSizeAll then
      seq {
        yield X, Y
      }
    else
      Prelude.undefined

  let private _forwardPropagate net reg (ts: TrainingState) X Y =
    let J', cache = Net.forwardPropagate { net with Parameters = ts.Parameters } X Y
    let J' = reg.RegularizeCost net.Parameters J'
    J', cache

  let private _backPropagate net reg cache =
    let gradients = Net.backPropagate net cache
    let gradients = reg.RegularizeGradients net.Parameters gradients
    gradients

  let private _trainNetworkFor1MiniBatch net reg hp m (_: Tensor<double>, ts: TrainingState, timer: Stopwatch) (X: Tensor<double>, Y: Tensor<double>): (Tensor<double> * TrainingState * Stopwatch) =
    timer.Start()
    let J', cache = _forwardPropagate net reg ts X Y
    let gradients = _backPropagate net reg cache
    let ts = Optimizer.updateParameters hp.LearningRate gradients ts
    timer.Stop()

    let J = (m |> TensorR0).PointwiseMultiply(J')
    J, ts, timer

  let private _trainNetworkFor1Epoch (timer: Stopwatch) (callback: EpochCallback) (net: Net) reg m (X: Tensor<double>) (Y: Tensor<double>) (hp: HyperParameters) (ts: TrainingState) epoch =
    timer.Restart()

    let J, ts, timer =
      (X, Y)
      |> _getMiniBatches hp.BatchSize
      |> Seq.fold (_trainNetworkFor1MiniBatch net reg hp m) (0. |> TensorR0, ts, timer)
    timer.Stop()

    let m = double X.ColumnCount
    let J = (TensorR0 (1. / m)).PointwiseMultiply(J)
    callback epoch timer.Elapsed J
    ts

  let trainWithGD (callback: EpochCallback) (net: Net) (X: Tensor<double>) (Y: Tensor<double>) (hp: HyperParameters) : Net =
    if X.ColumnCount <> Y.ColumnCount then
      failwithf "Column count of X (%d) is not the same as that of Y (%d)." X.ColumnCount Y.ColumnCount

    let timer = Stopwatch()

    let m = X.ColumnCount |> double
    let reg = Regularizer.get m hp.Regularizer
    let ts0 = Optimizer.initializeState net.Parameters hp.Optimizer

    let ts =
      seq { for epoch in 0 .. (hp.Epochs - 1) do epoch }
      |> Seq.fold (_trainNetworkFor1Epoch timer callback net reg m X Y hp) ts0

    { net with Parameters = ts.Parameters }
