namespace KS.FsDNN

[<AutoOpen>]
module OperationsDomain =

  let Scalar1 = TensorR0 1.

module Operations =

  module Add =
    let forwardPropagate (arg0: Tensor<double>) (arg1: Tensor<double>): Tensor<double> =
      arg0 + arg1

    let backPropagate (_: Cache<Tensor<double>>) _ (inG: Tensor<double>) =
      (inG, inG)

    let Functions : Op2Functions<_> = { F = forwardPropagate; B = backPropagate }

  module Multiply =
    let forwardPropagate (arg0: Tensor<double>) (arg1: Tensor<double>): Tensor<double> =
      arg0 * arg1

    let backPropagate (cache: Cache<Tensor<double>>) id (inG: Tensor<double>) =
      let arg0 = cache.[id].[0]
      let arg1 = cache.[id].[1]
      (inG.TransposeAndMultiply(arg1), arg0.TransposeThisAndMultiply(inG))

    let Functions : Op2Functions<_> = { F = forwardPropagate; B = backPropagate }

  module BinaryCrossEntropyLoss =
    let forwardPropagate (Y: Tensor<double>) (Ŷ: Tensor<double>): Tensor<double> =
      let c =
        Y.PointwiseMultiply(Ŷ.PointwiseLog()) +
        Y.Negate().Add(1.).PointwiseMultiply(Ŷ.Negate().Add(1.).PointwiseLog())

      let m = double Y.ColumnCount

      TensorR0 ((-1. / m) * c.Sum())

    let backPropagate (cache: Cache<Tensor<double>>) id (inG: Tensor<double>) =
      let g0 = inG
      let Y = cache.[id].[0]
      let Ŷ = cache.[id].[1]
      let m = double Y.ColumnCount
      let g1 = (Y.PointwiseDivide(Ŷ.Add(Constants.DivideBy0Guard)).Negate() + Y.Negate().Add(1.0).PointwiseDivide(Ŷ.Negate().Add(1.0 + Constants.DivideBy0Guard))).PointwiseDivide(TensorR0 m)
      (g0, inG.PointwiseMultiply(g1))

    let Functions : Op2Functions<_> = { F = forwardPropagate; B = backPropagate }

  module Sigmoid =
    let forwardPropagate (arg: Tensor<double>): Tensor<double> =
      arg.Negate().PointwiseExp().Add(1.0).PointwisePower(-1.0)

    let backPropagate (cache: Cache<Tensor<double>>) id (inG: Tensor<double>) =
      let arg = cache.[id].[0]
      let s = arg.Negate().PointwiseExp().Add(1.0).PointwisePower(-1.0)
      inG.PointwiseMultiply(s.PointwiseMultiply(s.Negate().Add(1.0)))

    let Functions : Op1Functions<_> = { F = forwardPropagate; B = backPropagate }
