﻿namespace KS.FsDNN

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
    let forwardPropagate (arg0: Tensor<double>) (arg1: Tensor<double>): Tensor<double> =
      arg0 + arg1

    let backPropagate (_: Cache<Tensor<double>>) _ (inG: Tensor<double>) =
      (inG, inG)

    let Definition: Operation2Definition<_> =
      { Name = "Add"
        Functions = { F = forwardPropagate; B = backPropagate } }

  module Multiply =
    let forwardPropagate (arg0: Tensor<double>) (arg1: Tensor<double>): Tensor<double> =
      arg0 * arg1

    let backPropagate (cache: Cache<Tensor<double>>) id (inG: Tensor<double>) =
      let arg0 = cache.[id].[0]
      let arg1 = cache.[id].[1]
      (inG.TransposeAndMultiply(arg1), arg0.TransposeThisAndMultiply(inG))

    let Definition: Operation2Definition<_> =
      { Name = "Multiply"
        Functions = { F = forwardPropagate; B = backPropagate } }

  module BCEWithLogitsLoss =
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

    let Definition: Operation2Definition<_> =
      { Name = "BCEWithLogitsLoss"
        Functions = { F = forwardPropagate; B = backPropagate } }

  module CCEWithLogitsLoss =
    let forwardPropagate (Y: Tensor<double>) (Ŷ: Tensor<double>): Tensor<double> =
      let exp = Ŷ.PointwiseExp()
      let softMax = exp / exp.ColumnSums()
      (Y.Negate().PointwiseMultiply(softMax.PointwiseLog())).Sum() |> TensorR0

    let backPropagate (cache: Cache<Tensor<double>>) id (inG: Tensor<double>) =
      let g0 = inG
      let Y = cache.[id].[0]
      let Ŷ = cache.[id].[1]
      // TODO: Get this from the cache
      let exp = Ŷ.PointwiseExp()
      let softMax = exp / exp.ColumnSums()
      let g1 = softMax - Y
      (g0, inG.PointwiseMultiply(g1))

    let Definition: Operation2Definition<_> =
      { Name = "CCEWithLogitsLoss"
        Functions = { F = forwardPropagate; B = backPropagate } }

  module MSELoss =
    let forwardPropagate (Y: Tensor<double>) (Ŷ: Tensor<double>): Tensor<double> =
      let c = (Ŷ - Y).PointwisePower(2.)
      let m = double Y.ColumnCount

      TensorR0 ((1. / (2. * m)) * c.Sum())

    let backPropagate (cache: Cache<Tensor<double>>) id (inG: Tensor<double>) =
      let g0 = inG
      let Y = cache.[id].[0]
      let Ŷ = cache.[id].[1]
      let m = double Y.ColumnCount
      let g1 = (Ŷ - Y).PointwiseDivide(TensorR0 m)
      (g0, inG.PointwiseMultiply(g1))

    let Definition: Operation2Definition<_> =
      { Name = "MSELoss"
        Functions = { F = forwardPropagate; B = backPropagate } }

  module Sigmoid =
    let forwardPropagate (arg: Tensor<double>): Tensor<double> =
      arg.Negate().PointwiseExp().Add(1.0).PointwisePower(-1.0)

    let backPropagate (cache: Cache<Tensor<double>>) id (inG: Tensor<double>) =
      let arg = cache.[id].[0]
      let s = arg.Negate().PointwiseExp().Add(1.0).PointwisePower(-1.0)
      inG.PointwiseMultiply(s.PointwiseMultiply(s.Negate().Add(1.0)))

    let Definition: Operation1Definition<_> =
      { Name = "Sigmoid"
        Functions = { F = forwardPropagate; B = backPropagate } }

  module Linear =
    let Definition: Operation1Definition<_> =
      { Name = "Linear"
        Functions = { F = id; B = fun _ _ -> id } }

  module HardMax =
    let forwardPropagate (arg: Tensor<double>): Tensor<double> =
      arg.ColumnHardMax()

    let Definition: Operation1Definition<_> =
      { Name = "HardMax"
        Functions = { F = forwardPropagate; B = fun _ _ -> Prelude.undefined } }
