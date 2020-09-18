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

[<Fact>]
let ``Simple forward`` () =
  let g = Op2 {| Id = "OpAdd"
                 Op = OpAdd
                 Arg0 = Arg {| Id = "Arg0"; Data = 1 |}
                 Arg1 = Arg {| Id = "Arg1"; Data = 2 |} |}

  let J, iValues = ComputationGraph.forward id forwardOp1 forwardOp2 g
  J |> should equal 3

  iValues |> Map.toList |> should equal [ "OpAdd", [| 1; 2 |] ]

[<Fact>]
let ``Complex forward`` () =
  let g = Op1 {| Id = "OpSquare"
                 Op = OpSquare
                 Arg =
                   Op2 {| Id = "OpMul"
                          Op = OpMul
                          Arg0 = Arg {| Id = "OpMul-Arg0"; Data = 4 |}
                          Arg1 =
                            Op2 {| Id = "OpAdd"
                                   Op = OpAdd
                                   Arg0 = Arg {| Id = "OpAdd-Arg0"; Data = 1 |}
                                   Arg1 = Arg {| Id = "OpAdd-Arg1"; Data = 2 |} |} |} |}

  let J, iValues = ComputationGraph.forward id forwardOp1 forwardOp2 g
  J |> should equal 144

  iValues |> Map.toList |> List.sortBy fst |> should equal [ ("OpAdd", [| 1; 2 |]); ("OpMul", [| 4; 3 |]); ("OpSquare", [| 12 |]) ]

let backPropagate1 fArg fOp1 fOp2 acc g =
  ComputationGraph.fold fArg fOp1 fOp2 acc g

let backPropArg1 (inG, iValues, gradients) id value  =
  (inG, iValues, gradients |> Map.add id inG)

let backPropOp11 (inG, iValues: Map<string, int[]>) id op =
  let outG =
    match op with
    | OpSquare ->
      inG * 2 * (iValues |> Map.find id).[0]

  outG

let backPropOp21 (inG, iValues: Map<string, int[]>) id op =
  let outG =
    match op with
    | OpAdd ->
      (inG, inG)
    | OpMul ->
      let outGs = iValues |> Map.find id
      (inG * outGs.[1], inG * outGs.[0])

  outG

[<Fact>]
let ``Simple back propagate`` () =
  let g = Op2 {| Id = "OpMul"
                 Op = OpMul
                 Arg0 = Arg {| Id = "Arg0"; Data = 1 |}
                 Arg1 = Arg {| Id = "Arg1"; Data = 2 |} |}

  let _, iValues = ComputationGraph.forward id forwardOp1 forwardOp2 g
  let g0', iValues', gradients' = ComputationGraph.backPropagate backPropArg1 backPropOp11 backPropOp21 (1, iValues, Map.empty) g

  g0' |> should equal 1
  iValues' |> should equal iValues
  gradients' |> Map.toList |> List.sortBy fst |> should equal [("Arg0", 2); ("Arg1", 1)]


[<Fact>]
let ``Complex back propagate`` () =
  let g = Op1 {| Id = "OpSquare"
                 Op = OpSquare
                 Arg =
                   Op2 {| Id = "OpMul"
                          Op = OpMul
                          Arg0 = Arg {| Id = "OpMul-Arg0"; Data = 4 |}
                          Arg1 =
                            Op2 {| Id = "OpAdd"
                                   Op = OpAdd
                                   Arg0 = Arg {| Id = "OpAdd-Arg0"; Data = 1 |}
                                   Arg1 = Arg {| Id = "OpAdd-Arg1"; Data = 2 |} |} |} |}

  let _, iValues = ComputationGraph.forward id forwardOp1 forwardOp2 g
  let g0', iValues', gradients' = ComputationGraph.backPropagate backPropArg1 backPropOp11 backPropOp21 (1, iValues, Map.empty) g

  g0' |> should equal 1
  iValues' |> should equal iValues
  gradients' |> Map.toList |> List.sortBy fst |> should equal [("OpAdd-Arg0", 96); ("OpAdd-Arg1", 96); ("OpMul-Arg0", 72)]
