namespace KS.FsDNN

[<AutoOpen>]
module OperationsDomain =

  let Scalar1 = TensorR0 1.

  type Operation1Definition<'a> =
    { Name: string
      Functions: Op1Functions<'a> }

  type Operation2Definition<'a> =
    { Name: string
      Functions: Op2Functions<'a> }

module Operations =

  module Add =
    let private _forwardPropagate (arg0: Tensor<double>) (arg1: Tensor<double>): Tensor<double> =
      arg0 + arg1

    let private _backPropagate (_: Cache<Tensor<double>>) _ (inG: Tensor<double>) =
      (inG, inG)

    let Definition: Operation2Definition<_> =
      { Name = "Add"
        Functions = { F = _forwardPropagate; B = _backPropagate } }

  module Multiply =
    let private _forwardPropagate (arg0: Tensor<double>) (arg1: Tensor<double>): Tensor<double> =
      arg0 * arg1

    let private _backPropagate (cache: Cache<Tensor<double>>) id (inG: Tensor<double>) =
      let arg0 = cache.[id].[0]
      let arg1 = cache.[id].[1]
      (inG.TransposeAndMultiply(arg1), arg0.TransposeThisAndMultiply(inG))

    let Definition: Operation2Definition<_> =
      { Name = "Multiply"
        Functions = { F = _forwardPropagate; B = _backPropagate } }
