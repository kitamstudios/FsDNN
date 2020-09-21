namespace KS.FsDNN

[<AutoOpen>]
module OperationsDomain =

  let Scalar1 = Tensor.ofListOfList [[1.]]

  type Operations1 =
    | OpSigmoid

  type Operations2 =
    | OpAdd
    | OpMultiply
    | OpCrossEntropyLoss

module Operations =

  module Add =
    let forwardPropagate (arg0: Tensor<double>) (arg1: Tensor<double>): Tensor<double> =
      arg0 + arg1

    let backPropagate (inG: Tensor<double>) (_: Tensor<double>) (_: Tensor<double>) =
      (inG, inG)

  module Multiply =
    let forwardPropagate (arg0: Tensor<double>) (arg1: Tensor<double>): Tensor<double> =
      arg0 * arg1

    let backPropagate (inG: Tensor<double>) (arg0: Tensor<double>) (arg1: Tensor<double>) =
      (inG.TransposeAndMultiply(arg1), arg0.TransposeThisAndMultiply(inG))

  module BinaryCrossEntropyLoss =
    let forwardPropagate (Y: Tensor<double>) (Ŷ: Tensor<double>): Tensor<double> =
      let c =
        Y.PointwiseMultiply(Ŷ.PointwiseLog()) +
        Y.Negate().Add(1.).PointwiseMultiply(Ŷ.Negate().Add(1.).PointwiseLog())

      let m = double Y.ColumnCount

      R0 ((-1. / m) * c.Sum())

    let backPropagate (inG: Tensor<double>) (Y: Tensor<double>) (Ŷ: Tensor<double>) =
      let g0 = inG
      let g1 = (Y.PointwiseDivide(Ŷ.Add(Constants.DivideBy0Guard)).Negate() + Y.Negate().Add(1.0).PointwiseDivide(Ŷ.Negate().Add(1.0 + Constants.DivideBy0Guard)))
      (g0, g1.PointwiseMultiply(inG))

  module Sigmoid =
    let forwardPropagate (arg: Tensor<double>): Tensor<double> =
      arg.Negate().PointwiseExp().Add(1.0).PointwisePower(-1.0)

    let backPropagate (inG: Tensor<double>) (arg: Tensor<double>) =
      let s = arg.Negate().PointwiseExp().Add(1.0).PointwisePower(-1.0)
      inG.PointwiseMultiply(s.PointwiseMultiply(s.Negate().Add(1.0)))

  let forwardArg parameters id: Tensor<double> =
    parameters |> Map.find id

  let forwardOp1 o (arg: Tensor<double>): Tensor<double> =
    match o with
    | OpSigmoid -> Sigmoid.forwardPropagate arg

  let forwardOp2 o (arg0: Tensor<double>) (arg1: Tensor<double>): Tensor<double> =
    match o with
    | OpAdd -> Add.forwardPropagate arg0 arg1
    | OpMultiply -> Multiply.forwardPropagate arg0 arg1
    | OpCrossEntropyLoss -> BinaryCrossEntropyLoss.forwardPropagate arg0 arg1

  let backPropagateOp1 (iValues: Map<string, Tensor<double>[]>) inG id op =
    match op with
    | OpSigmoid -> Sigmoid.backPropagate inG iValues.[id].[0]

  let backPropagateOp2 (iValues: Map<string, Tensor<double>[]>) inG id op =
    let args = iValues.[id]
    match op with
    | OpAdd -> Add.backPropagate inG args.[0] args.[1]
    | OpMultiply -> Multiply.backPropagate inG args.[0] args.[1]
    | OpCrossEntropyLoss -> BinaryCrossEntropyLoss.backPropagate inG args.[0] args.[1]
