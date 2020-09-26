namespace KS.FsDNN

module NullOptimizerDomain =
  ()

module NullOptimizer =
  let updateParameters (lr: Tensor<double>) (parameters: Parameters) (gradients: Gradients) =
    parameters
    |> Map.map (fun id value -> value - (lr * gradients.[id]))

module MomentumOptimizerDomain =
  type MomentumParameters =
    { Beta: double }
    static member Defaults: MomentumParameters =
      { Beta = 0.9 }

  /// Exponentially weighted average of the gradients.
  type GradientVelocity = { dWv: Tensor<double>; dbv: Tensor<double> }
  type GradientVelocities = Map<int, GradientVelocity>

module MomentumOptimizer =
  ()

module AdaMOptimizerDomain =
  type AdaMParameters =
    { Beta1: double; Beta2: double; Epsilon: double }
    static member Defaults: AdaMParameters =
      { Beta1 = 0.9; Beta2 = 0.999; Epsilon = 1e-8 }

  /// Exponentially weighted average of the squared gradient.
  type SquaredGradientVelocity = { dWs: Tensor<double>; dbs: Tensor<double> }
  type SquaredGradientVelocities = Map<int, SquaredGradientVelocity>

module AdaMOptimizer =
  ()

[<AutoOpen>]
module OptimizerDomain =
  open MomentumOptimizerDomain
  open AdaMOptimizerDomain

  type Optimizer =
    | NullOptimizer
    | MomentumOptimizer of MomentumParameters
    | AdaMOptimizer of AdaMParameters

  type TrainingState =
    | NullOptimizerState of Parameters
    | MomentumOptimizerState of Parameters * GradientVelocities
    | AdaMOptimizerState of Parameters * GradientVelocities * SquaredGradientVelocities * double
    with
    member this.Parameters =
      match this with
      | NullOptimizerState p -> p
      | MomentumOptimizerState (p, _) -> p
      | AdaMOptimizerState (p, _, _, _) -> p

module Optimizer =

  let initializeState optimizer parameters =
    match optimizer with
    | NullOptimizer -> NullOptimizerState parameters
    | _ -> Prelude.undefined

  let updateParameters lr (ts: TrainingState) (gradients: Gradients): TrainingState =
    match ts with
    | NullOptimizerState parameters -> NullOptimizer.updateParameters lr parameters gradients |> NullOptimizerState
    | _ -> Prelude.undefined
