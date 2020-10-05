namespace KS.FsDNN

open System
open System.Diagnostics

[<AutoOpen>]
module TrainerDomain =
  type EpochCallback = int -> TimeSpan -> Tensor<double> -> unit

  type HyperParameters =
    { Epochs : int
      LearningRate: Tensor<double>
      Regularizer: Regularizer
      Optimizer: Optimizer
      BatchSize: BatchSize } with
    static member Defaults =
      { Epochs = 1_000
        LearningRate = TensorR0 0.01
        Regularizer = L2Regularizer 0.7
        Optimizer = AdaMOptimizer AdaMOptimizerDomain.AdaMParameters.Defaults
        BatchSize = BatchSizeAll (* BatchSize64 *) }

module Trainer =

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

    let J = (m |> double |> TensorR0).PointwiseMultiply(J')
    J, ts, timer

  let private _trainNetworkFor1Epoch (timer: Stopwatch) (callback: EpochCallback) (net: Net) reg m (batches: (Tensor<double> * Tensor<double>) seq) (hp: HyperParameters) (ts: TrainingState) epoch =
    timer.Restart()

    let J, ts, timer =
      batches
      |> Seq.fold (_trainNetworkFor1MiniBatch net reg hp m) (0. |> TensorR0, ts, timer)
    timer.Stop()

    let J = (TensorR0 (1. / double m)).PointwiseMultiply(J)
    callback epoch timer.Elapsed J
    ts

  let trainWithGD (callback: EpochCallback) (net: Net) (batches: (Tensor<double> * Tensor<double>) seq) (m: int) (hp: HyperParameters) : Net =

    let timer = Stopwatch()

    let reg = Regularizer.get m hp.Regularizer
    let ts0 = Optimizer.initializeState net.Parameters hp.Optimizer

    let ts =
      seq { for epoch in 0 .. (hp.Epochs - 1) do epoch }
      |> Seq.fold (_trainNetworkFor1Epoch timer callback net reg m batches hp) ts0

    { net with Parameters = ts.Parameters }
