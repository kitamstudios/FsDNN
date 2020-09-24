namespace KS.FsDNN

[<AutoOpen>]
module OperationsDomain =

  let Scalar1 = TensorR0 1.

module Operations =

  module Add =
    let private _forwardPropagate (arg0: Tensor<double>) (arg1: Tensor<double>) =
      [| arg0 + arg1; arg0; arg1 |]

    let private _backPropagate (_: Cache<Tensor<double>>) _ (inG: Tensor<double>) =
      (inG, inG)

    let Definition: Operation2Definition<_> =
      { Name = "Add"
        Functions = { F = _forwardPropagate; B = _backPropagate } }

  module Multiply =
    let private _forwardPropagate (arg0: Tensor<double>) (arg1: Tensor<double>)=
      [| arg0 * arg1; arg0; arg1 |]

    let private _backPropagate (cache: Cache<Tensor<double>>) id (inG: Tensor<double>) =
      let arg0 = cache.[id].[1]
      let arg1 = cache.[id].[2]
      (inG.TransposeAndMultiply(arg1), arg0.TransposeThisAndMultiply(inG))

    let Definition: Operation2Definition<_> =
      { Name = "Multiply"
        Functions = { F = _forwardPropagate; B = _backPropagate } }
