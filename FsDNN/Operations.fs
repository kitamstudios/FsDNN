namespace KS.FsDNN

[<AutoOpen>]
module OperationsDomain =

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

    let backPropagate (inG: Tensor<double>) (arg0: Tensor<double>) (arg1: Tensor<double>) =
      (inG, inG)

  module Multiply =
    let forwardPropagate (arg0: Tensor<double>) (arg1: Tensor<double>): Tensor<double> =
      arg0 * arg1

    let backPropagate (inG: Tensor<double>) (arg0: Tensor<double>) (arg1: Tensor<double>) =
      (inG, inG)

  module CrossEntropyLoss =
    let forwardPropagate (arg0: Tensor<double>) (arg1: Tensor<double>): Tensor<double> =
      Prelude.undefined

    let backPropagate (inG: Tensor<double>) (arg0: Tensor<double>) (arg1: Tensor<double>) =
      (inG, inG)

  module Sigmoid =
    let forwardPropagate (arg: Tensor<double>): Tensor<double> =
      arg.Negate().PointwiseExp().Add(1.0).PointwisePower(-1.0)

    let backPropagate (inG: Tensor<double>) (arg: Tensor<double>) =
      inG

  let forwardArg parameters id: Tensor<double> =
    parameters |> Map.find id

  let forwardOp1 o (arg: Tensor<double>): Tensor<double> =
    match o with
    | OpSigmoid -> Sigmoid.forwardPropagate arg

  let forwardOp2 o (arg0: Tensor<double>) (arg1: Tensor<double>): Tensor<double> =
    match o with
    | OpAdd -> Add.forwardPropagate arg0 arg1
    | OpMultiply -> Multiply.forwardPropagate arg0 arg1
    | OpCrossEntropyLoss -> CrossEntropyLoss.forwardPropagate arg0 arg1

  let backPropagateOp1 (iValues: Map<string, Tensor<double>[]>) inG id op =
    match op with
    | OpSigmoid -> Sigmoid.backPropagate inG iValues.[id].[0]

  let backPropagateOp2 (iValues: Map<string, Tensor<double>[]>) inG id op =
    let args = iValues.[id]
    match op with
    | OpAdd -> Add.backPropagate inG args.[0] args.[1]
    | OpMultiply -> Multiply.backPropagate inG args.[0] args.[1]
    | OpCrossEntropyLoss -> CrossEntropyLoss.backPropagate inG args.[0] args.[1]
