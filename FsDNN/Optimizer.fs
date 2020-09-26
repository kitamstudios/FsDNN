namespace KS.FsDNN

module NullOptimizerDomain =

  type NullOptimizerState = Parameters

module NullOptimizer =
  open NullOptimizerDomain

  let initialize = id

  let updateParameters (lr: Tensor<double>) (gradients: Gradients) (s: NullOptimizerState): NullOptimizerState =
    s
    |> Map.map (fun id value -> value - (lr * gradients.[id]))

module MomentumOptimizerDomain =
  type MomentumParameters =
    { Beta: double }
    static member Defaults: MomentumParameters =
      { Beta = 0.9 }

  /// Exponentially weighted average of the gradients.
  type GradientVelocity = { dWv: Tensor<double>; dbv: Tensor<double> }
  type GradientVelocities = Map<int, GradientVelocity>

  type MomentumOptimizerState = Parameters * GradientVelocities

module MomentumOptimizer =
  open MomentumOptimizerDomain

  let initialize parameters =
    Prelude.undefined

  let updateParameters (lr: Tensor<double>) (gradients: Gradients) (s: MomentumOptimizerState): MomentumOptimizerState =
    s

module AdaMOptimizerDomain =
  type AdaMParameters =
    { Beta1: double; Beta2: double; Epsilon: double }
    static member Defaults: AdaMParameters =
      { Beta1 = 0.9; Beta2 = 0.999; Epsilon = 1e-8 }

  /// Exponentially weighted average of the squared gradient.
  type SquaredGradientVelocity = { dWs: Tensor<double>; dbs: Tensor<double> }
  type SquaredGradientVelocities = Map<int, SquaredGradientVelocity>

  type AdaMOptimizerState = Parameters * MomentumOptimizerDomain.GradientVelocities * SquaredGradientVelocity * double

module AdaMOptimizer =
  open AdaMOptimizerDomain

  let initialize parameters =
    Prelude.undefined

  let updateParameters (lr: Tensor<double>) (gradients: Gradients) (s: AdaMOptimizerState): AdaMOptimizerState =
    s

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
    | MomentumOptimizerState of MomentumOptimizerDomain.MomentumOptimizerState
    | AdaMOptimizerState of AdaMOptimizerDomain.AdaMOptimizerState
    with
    member this.Parameters =
      match this with
      | NullOptimizerState p -> p
      | MomentumOptimizerState (p, _) -> p
      | AdaMOptimizerState (p, _, _, _) -> p

module Optimizer =

  let initializeState parameters = function
    | NullOptimizer -> NullOptimizer.initialize parameters |> NullOptimizerState
    | MomentumOptimizer _ -> MomentumOptimizer.initialize parameters |> MomentumOptimizerState
    | AdaMOptimizer _ -> MomentumOptimizer.initialize parameters |> AdaMOptimizerState

  let updateParameters lr (gradients: Gradients) = function
    | NullOptimizerState s -> NullOptimizer.updateParameters lr gradients s |> NullOptimizerState
    | MomentumOptimizerState s -> MomentumOptimizer.updateParameters lr gradients s |> MomentumOptimizerState
    | AdaMOptimizerState s -> AdaMOptimizer.updateParameters lr gradients s |> AdaMOptimizerState
