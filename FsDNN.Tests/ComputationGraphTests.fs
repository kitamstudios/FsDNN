module KS.FsDNN.Tests.ComputationGraph

open KS.FsDNN
open Xunit
open FsUnit.Xunit

type Operations1 =
  | OpSquare

type Operations2 =
  | OpAdd
  | OpMul

let forwardArg parameters id: int =
  parameters |> Map.find id

let forwardOp1 o arg: int =
  match o with
  | OpSquare ->
    arg * arg

let forwardOp2 o arg0 arg1: int =
  match o with
  | OpAdd -> arg0 + arg1
  | OpMul -> arg0 * arg1

[<Fact>]
let ``Simple predict`` () =
  let g = Op2 {| Id = "OpAdd"
                 Op = OpAdd
                 Arg0 = Arg {| Id = "Arg0"; TrackGradient = true |}
                 Arg1 = Arg {| Id = "Arg1"; TrackGradient = true |} |}

  let parameters = Map.empty |> Map.add "Arg0" 1 |> Map.add "Arg1" 2

  let Y = ComputationGraph.predict (forwardArg parameters) forwardOp1 forwardOp2 g
  Y |> should equal 3

[<Fact>]
let ``Simple forward`` () =
  let g = Op2 {| Id = "OpAdd"
                 Op = OpAdd
                 Arg0 = Arg {| Id = "Arg0"; TrackGradient = true |}
                 Arg1 = Arg {| Id = "Arg1"; TrackGradient = true |} |}

  let parameters = Map.empty |> Map.add "Arg0" 1 |> Map.add "Arg1" 2

  let J, iValues = ComputationGraph.forward (forwardArg parameters) forwardOp1 forwardOp2 g
  J |> should equal 3

  iValues |> Map.toList |> should equal [ "OpAdd", [| 1; 2 |] ]

[<Fact>]
let ``Complex predict`` () =
  let g = Op1 {| Id = "OpSquare"
                 Op = OpSquare
                 Arg =
                   Op2 {| Id = "OpMul"
                          Op = OpMul
                          Arg0 = Arg {| Id = "OpMul-Arg0"; TrackGradient = true |}
                          Arg1 =
                            Op2 {| Id = "OpAdd"
                                   Op = OpAdd
                                   Arg0 = Arg {| Id = "OpAdd-Arg0"; TrackGradient = true |}
                                   Arg1 = Arg {| Id = "OpAdd-Arg1"; TrackGradient = true |} |} |} |}

  let parameters = Map.empty |> Map.add "OpMul-Arg0" 4 |> Map.add "OpAdd-Arg0" 1 |> Map.add "OpAdd-Arg1" 2

  let Y = ComputationGraph.predict (forwardArg parameters) forwardOp1 forwardOp2 g
  Y |> should equal 144

[<Fact>]
let ``Complex forward`` () =
  let g = Op1 {| Id = "OpSquare"
                 Op = OpSquare
                 Arg =
                   Op2 {| Id = "OpMul"
                          Op = OpMul
                          Arg0 = Arg {| Id = "OpMul-Arg0"; TrackGradient = true |}
                          Arg1 =
                            Op2 {| Id = "OpAdd"
                                   Op = OpAdd
                                   Arg0 = Arg {| Id = "OpAdd-Arg0"; TrackGradient = true |}
                                   Arg1 = Arg {| Id = "OpAdd-Arg1"; TrackGradient = true |} |} |} |}

  let parameters = Map.empty |> Map.add "OpMul-Arg0" 4 |> Map.add "OpAdd-Arg0" 1 |> Map.add "OpAdd-Arg1" 2

  let J, iValues = ComputationGraph.forward (forwardArg parameters) forwardOp1 forwardOp2 g
  J |> should equal 144

  iValues |> Map.toList |> List.sortBy fst |> should equal [ ("OpAdd", [| 1; 2 |]); ("OpMul", [| 4; 3 |]); ("OpSquare", [| 12 |]) ]

let backPropOp1 (iValues: Map<string, int[]>) inG id op =
  match op with
  | OpSquare ->
    inG * 2 * (iValues |> Map.find id).[0]

let backPropOp2 (iValues: Map<string, int[]>) inG id op =
  match op with
  | OpAdd ->
    (inG, inG)
  | OpMul ->
    let outGs = iValues |> Map.find id
    (inG * outGs.[1], inG * outGs.[0])

[<Fact>]
let ``Simple back propagate`` () =
  let g = Op2 {| Id = "OpMul"
                 Op = OpMul
                 Arg0 = Arg {| Id = "Arg0"; TrackGradient = true |}
                 Arg1 = Arg {| Id = "Arg1"; TrackGradient = true |} |}

  let parameters = Map.empty |> Map.add "Arg0" 1 |> Map.add "Arg1" 2

  let _, iValues = ComputationGraph.forward (forwardArg parameters)  forwardOp1 forwardOp2 g
  let gradients = ComputationGraph.backPropagate (backPropOp1 iValues) (backPropOp2 iValues) 1 g

  gradients |> Map.toList |> List.sortBy fst |> should equal [("Arg0", 2); ("Arg1", 1)]


[<Fact>]
let ``Simple back propagate - without gradient tracking`` () =
  let g = Op2 {| Id = "OpMul"
                 Op = OpMul
                 Arg0 = Arg {| Id = "Arg0"; TrackGradient = false |}
                 Arg1 = Arg {| Id = "Arg1"; TrackGradient = true |} |}

  let parameters = Map.empty |> Map.add "Arg0" 1 |> Map.add "Arg1" 2

  let _, iValues = ComputationGraph.forward (forwardArg parameters)  forwardOp1 forwardOp2 g
  let gradients = ComputationGraph.backPropagate (backPropOp1 iValues) (backPropOp2 iValues) 1 g

  gradients |> Map.toList |> List.sortBy fst |> should equal [("Arg1", 1)]


[<Fact>]
let ``Complex back propagate`` () =
  let g = Op1 {| Id = "OpSquare"
                 Op = OpSquare
                 Arg =
                   Op2 {| Id = "OpMul"
                          Op = OpMul
                          Arg0 = Arg {| Id = "OpMul-Arg0"; TrackGradient = true |}
                          Arg1 =
                            Op2 {| Id = "OpAdd"
                                   Op = OpAdd
                                   Arg0 = Arg {| Id = "OpAdd-Arg0"; TrackGradient = true |}
                                   Arg1 = Arg {| Id = "OpAdd-Arg1"; TrackGradient = true |} |} |} |}

  let parameters = Map.empty |> Map.add "OpMul-Arg0" 4 |> Map.add "OpAdd-Arg0" 1 |> Map.add "OpAdd-Arg1" 2
  let _, iValues = ComputationGraph.forward (forwardArg parameters) forwardOp1 forwardOp2 g
  let gradients = ComputationGraph.backPropagate (backPropOp1 iValues) (backPropOp2 iValues) 1 g

  gradients |> Map.toList |> List.sortBy fst |> should equal [("OpAdd-Arg0", 96); ("OpAdd-Arg1", 96); ("OpMul-Arg0", 72)]
