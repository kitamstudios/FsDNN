namespace KS.FsDNN

module NullOptimizerDomain =

  type NullOptimizerState = Parameters

module NullOptimizer =
  open NullOptimizerDomain

  let initializeState = id

  let updateParameters (lr: Tensor<double>) (gradients: Gradients) (s: NullOptimizerState): NullOptimizerState =
    s
    |> Map.map (fun id value -> value - (lr * gradients.[id]))

module MomentumOptimizerDomain =

  type MomentumParameters =
    { Beta: double }
    static member Defaults: MomentumParameters =
      { Beta = 0.9 }

  type MomentumOptimizerState = Parameters * GradientVelocities

module MomentumOptimizer =
  open MomentumOptimizerDomain

  let initializeState (parameters: Parameters): MomentumOptimizerState =
    let gv = parameters |> Map.map (fun _ v -> v.CreateShapeSameAsWithZeros())
    parameters, gv

  let updateParameters (mp: MomentumParameters) (lr: Tensor<double>) (gradients: Gradients) ((p, gv): MomentumOptimizerState): MomentumOptimizerState =
    let _folder (pAcc, vAcc) (k: string) (grad: Tensor<double>) =
      let gv = mp.Beta * gv.[k] + (1. - mp.Beta) * grad
      let p = p.[k] - lr * gv

      (pAcc |> Map.add k p), (vAcc |> Map.add k gv)

    let p, gv =
      gradients
      |> Map.fold _folder (Map.empty, Map.empty)

    (p, gv) |> MomentumOptimizerState

module AdaMOptimizerDomain =

  type AdaMParameters =
    { Beta1: double; Beta2: double; Epsilon: double }
    static member Defaults: AdaMParameters =
      { Beta1 = 0.9; Beta2 = 0.999; Epsilon = 1e-8 }

  type AdaMOptimizerState = Parameters * GradientVelocities * SquaredGradientVelocities * double

module AdaMOptimizer =
  open AdaMOptimizerDomain

  let initializeState (parameters: Parameters): AdaMOptimizerState =
    let gv = parameters |> Map.map (fun _ v -> v.CreateShapeSameAsWithZeros())
    let sgv = parameters |> Map.map (fun _ v -> v.CreateShapeSameAsWithZeros())
    parameters, gv, sgv, 1.

  let updateParameters (ap: AdaMParameters) (lr: Tensor<double>) (gradients: Gradients) ((p, gv, sgv, t): AdaMOptimizerState): AdaMOptimizerState =
    let _folder (pAcc, vAcc, sAcc) (k: string) (grad: Tensor<double>) =
      let gv = ap.Beta1 * gv.[k] + (1. - ap.Beta1) * grad

      let gv_corrected = gv / (1. - ap.Beta1 ** t)
      let sgv = ap.Beta2 * sgv.[k] + (1. - ap.Beta2) * grad.PointwisePower(2.)

      let sgv_corrected = sgv / (1. - ap.Beta2 ** t)
      let p = p.[k] - lr * gv_corrected.PointwiseDivide(sgv_corrected.PointwisePower(0.5) + ap.Epsilon)

      (pAcc |> Map.add k p), (vAcc |> Map.add k gv), (sAcc |> Map.add k sgv)

    let p, gv, sgv =
      gradients
      |> Map.fold _folder (Map.empty, Map.empty, Map.empty)

    (p, gv, sgv, t + 1.) |> AdaMOptimizerState

[<AutoOpen>]
module OptimizerDomain =
  open NullOptimizerDomain
  open MomentumOptimizerDomain
  open AdaMOptimizerDomain

  type Optimizer =
    | NullOptimizer
    | MomentumOptimizer of MomentumParameters
    | AdaMOptimizer of AdaMParameters

  type TrainingState =
    | NullOptimizerState of NullOptimizerState
    | MomentumOptimizerState of MomentumParameters * MomentumOptimizerState
    | AdaMOptimizerState of AdaMParameters * AdaMOptimizerState
    with
    member this.Parameters =
      match this with
      | NullOptimizerState p -> p
      | MomentumOptimizerState (_, (p, _)) -> p
      | AdaMOptimizerState (_, (p, _, _, _)) -> p

module Optimizer =

  let initializeState parameters = function
    | NullOptimizer -> NullOptimizer.initializeState parameters |> NullOptimizerState
    | MomentumOptimizer mp -> (mp, MomentumOptimizer.initializeState parameters) |> MomentumOptimizerState
    | AdaMOptimizer ap -> (ap, AdaMOptimizer.initializeState parameters) |> AdaMOptimizerState

  let updateParameters lr (gradients: Gradients) = function
    | NullOptimizerState s -> NullOptimizer.updateParameters lr gradients s |> NullOptimizerState
    | MomentumOptimizerState (mp, s) -> (mp, MomentumOptimizer.updateParameters mp lr gradients s) |> MomentumOptimizerState
    | AdaMOptimizerState (ap, s)  -> (ap, AdaMOptimizer.updateParameters ap lr gradients s) |> AdaMOptimizerState
