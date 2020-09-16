module KS.FsDNN.Tests.ComputationGraph

open KS.FsDNN
open Xunit
open FsUnit.Xunit

type Operations1 =
  | OpSquare

type Operations2 =
  | OpAdd
  | OpMul

let forwardArg arg : int =
  arg.Data

let forwardOp1 o arg : int =
  match o with
  | OpSquare ->
    arg * arg

let forwardOp2 o arg0 arg1 : int =
  match o with
  | OpAdd -> arg0 + arg1
  | OpMul -> arg0 * arg1

let getIntermediateValues = ComputationGraph.fold (fun acc e -> e.Data :: acc) (fun acc o -> o.IVal.Data :: acc) (fun acc o -> o.IVal.Data :: o.Arg0.Data :: acc) []

[<Fact>]
let ``Simple forward`` () =
  let g = Op2 { Op = OpAdd
                IVal = { Data = 0; Gradient = 0 }
                In0 = 0
                In1 = 0
                Arg0 = { Data = 1; Gradient = 0 }
                Arg1 = Arg { Data = 2; Gradient = 0 } }

  let J = ComputationGraph.forward forwardArg forwardOp1 forwardOp2 g
  J |> should equal 3

  let intermediates = getIntermediateValues g
  intermediates |> should equal [2; 3; 1]

[<Fact>]
let ``Complex forward`` () =
  let g = Op1 { Op = OpSquare
                IVal = { Data = 0; Gradient = 0 }
                In = 0
                Arg =
                  Op2 { Op = OpMul
                        IVal = { Data = 0; Gradient = 0 }
                        In0 = 0
                        In1 = 0
                        Arg0 = { Data = 4; Gradient = 0 }
                        Arg1 =
                          Op2 { Op = OpAdd
                                IVal = { Data = 0; Gradient = 0 }
                                In0 = 0
                                In1 = 0
                                Arg0 = { Data = 1; Gradient = 0 }
                                Arg1 = Arg { Data = 2; Gradient = 0 } } } }

  let J = ComputationGraph.forward forwardArg forwardOp1 forwardOp2 g
  J |> should equal 144

  let intermediates = getIntermediateValues g
  intermediates |> should equal [2; 3; 1; 12; 4; 144]

let backPropArg acc a =
  a.Gradient <- acc
  acc

let backPropOp1 acc (o: Op1Info<_, _, _>)  =
  let acc' =
    match o.Op with
    | OpSquare ->
      acc * 2 * o.In
  acc'

let backPropOp2 acc (o: Op2Info<_, _, _>)  =
  let acc' =
    match o.Op with
    | OpAdd ->
      o.Arg0.Gradient <- acc
      acc
    | OpMul ->
      o.Arg0.Gradient <- acc * o.In1
      acc * o.In0
  acc'

let getGradients = ComputationGraph.fold (fun acc e -> e.Gradient :: acc) (fun acc _ -> acc) (fun acc o -> o.Arg0.Gradient :: acc) []

[<Fact>]
let ``Simple back propagate`` () =
  let g = Op2 { Op = OpMul
                In0 = 0
                In1 = 0
                IVal = { Data = 0; Gradient = 0 }
                Arg0 = { Data = 1; Gradient = 0 }
                Arg1 = Arg { Data = 2; Gradient = 0 } }

  ComputationGraph.forward forwardArg forwardOp1 forwardOp2 g |> ignore
  ComputationGraph.backPropagate backPropArg backPropOp1 backPropOp2 1 g |> ignore

  let gradients = getGradients g

  gradients |> should equal [1; 2]

[<Fact>]
let ``Complex back propagate`` () =
  let g = Op1 { Op = OpSquare
                IVal = { Data = 0; Gradient = 0 }
                In = 0
                Arg =
                  Op2 { Op = OpMul
                        IVal = { Data = 0; Gradient = 0 }
                        In0 = 0
                        In1 = 0
                        Arg0 = { Data = 4; Gradient = 0 }
                        Arg1 =
                          Op2 { Op = OpAdd
                                IVal = { Data = 0; Gradient = 0 }
                                In0 = 0
                                In1 = 0
                                Arg0 = { Data = 1; Gradient = 0 }
                                Arg1 = Arg { Data = 2; Gradient = 0 } } } }

  ComputationGraph.forward forwardArg forwardOp1 forwardOp2 g |> ignore
  ComputationGraph.backPropagate backPropArg backPropOp1 backPropOp2 1 g |> ignore

  let gradients = getGradients g

  gradients |> should equal [96; 96; 72]
