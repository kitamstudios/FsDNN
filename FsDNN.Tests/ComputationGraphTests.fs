module KS.FsDNN.Tests.ComputationGraph

open KS.FsDNN
open Xunit
open FsUnit.Xunit

let forwardArg parameters id: int =
  parameters |> Map.find id

module Operations =
  module Add =
    let f arg0 arg1 =
      arg0 + arg1

    let b _ _ inG =
      (inG, inG)

    let Functions : Op2Functions<_> = { F = f; B = b }

  module Multiply =
    let f arg0 arg1 =
      arg0 * arg1

    let b (cache: Cache<int>) id inG =
      let outGs = cache |> Map.find id
      (inG * outGs.[1], inG * outGs.[0])

    let Functions : Op2Functions<_> = { F = f; B = b }

  module Square =
    let f arg =
      arg * arg

    let b (cache: Cache<int>) id inG =
      inG * 2 * (cache |> Map.find id).[0]

    let Functions : Op1Functions<_> = { F = f; B = b }

[<Fact>]
let ``Simple predict`` () =
  let g = Op2 {| Id = "OpAdd"
                 Functions = Operations.Add.Functions
                 Arg0 = Arg {| Id = "Arg0"; TrackGradient = true |}
                 Arg1 = Arg {| Id = "Arg1"; TrackGradient = true |} |}

  let parameters = Map.empty |> Map.add "Arg0" 1 |> Map.add "Arg1" 2

  let Y = ComputationGraph.predict (forwardArg parameters) g
  Y |> should equal 3

[<Fact>]
let ``Simple forward`` () =
  let g = Op2 {| Id = "OpAdd"
                 Functions = Operations.Add.Functions
                 Arg0 = Arg {| Id = "Arg0"; TrackGradient = true |}
                 Arg1 = Arg {| Id = "Arg1"; TrackGradient = true |} |}

  let parameters = Map.empty |> Map.add "Arg0" 1 |> Map.add "Arg1" 2

  let J, cache = ComputationGraph.forward (forwardArg parameters) g
  J |> should equal 3

  cache |> Map.toList |> should equal [ "OpAdd", [| 1; 2 |] ]

[<Fact>]
let ``Complex predict`` () =
  let g = Op1 {| Id = "OpSquare"
                 Functions = Operations.Square.Functions
                 Arg =
                   Op2 {| Id = "OpMul"
                          Functions = Operations.Multiply.Functions
                          Arg0 = Arg {| Id = "OpMul-Arg0"; TrackGradient = true |}
                          Arg1 =
                            Op2 {| Id = "OpAdd"
                                   Functions = Operations.Add.Functions
                                   Arg0 = Arg {| Id = "OpAdd-Arg0"; TrackGradient = true |}
                                   Arg1 = Arg {| Id = "OpAdd-Arg1"; TrackGradient = true |} |} |} |}

  let parameters = Map.empty |> Map.add "OpMul-Arg0" 4 |> Map.add "OpAdd-Arg0" 1 |> Map.add "OpAdd-Arg1" 2

  let Y = ComputationGraph.predict (forwardArg parameters) g
  Y |> should equal 144

[<Fact>]
let ``Complex forward`` () =
  let g = Op1 {| Id = "OpSquare"
                 Functions = Operations.Square.Functions
                 Arg =
                   Op2 {| Id = "OpMul"
                          Functions = Operations.Multiply.Functions
                          Arg0 = Arg {| Id = "OpMul-Arg0"; TrackGradient = true |}
                          Arg1 =
                            Op2 {| Id = "OpAdd"
                                   Functions = Operations.Add.Functions
                                   Arg0 = Arg {| Id = "OpAdd-Arg0"; TrackGradient = true |}
                                   Arg1 = Arg {| Id = "OpAdd-Arg1"; TrackGradient = true |} |} |} |}

  let parameters = Map.empty |> Map.add "OpMul-Arg0" 4 |> Map.add "OpAdd-Arg0" 1 |> Map.add "OpAdd-Arg1" 2

  let J, cache = ComputationGraph.forward (forwardArg parameters) g
  J |> should equal 144

  cache |> Map.toList |> List.sortBy fst |> should equal [ ("OpAdd", [| 1; 2 |]); ("OpMul", [| 4; 3 |]); ("OpSquare", [| 12 |]) ]

[<Fact>]
let ``Simple back propagate`` () =
  let g = Op2 {| Id = "OpMul"
                 Functions = Operations.Multiply.Functions
                 Arg0 = Arg {| Id = "Arg0"; TrackGradient = true |}
                 Arg1 = Arg {| Id = "Arg1"; TrackGradient = true |} |}

  let parameters = Map.empty |> Map.add "Arg0" 1 |> Map.add "Arg1" 2

  let _, cache = ComputationGraph.forward (forwardArg parameters)  g
  let gradients = ComputationGraph.backPropagate (fun x -> x cache) (fun x -> x cache) 1 g

  gradients |> Map.toList |> List.sortBy fst |> should equal [("Arg0", 2); ("Arg1", 1)]


[<Fact>]
let ``Simple back propagate - without gradient tracking`` () =
  let g = Op2 {| Id = "OpMul"
                 Functions = Operations.Multiply.Functions
                 Arg0 = Arg {| Id = "Arg0"; TrackGradient = false |}
                 Arg1 = Arg {| Id = "Arg1"; TrackGradient = true |} |}

  let parameters = Map.empty |> Map.add "Arg0" 1 |> Map.add "Arg1" 2

  let _, cache = ComputationGraph.forward (forwardArg parameters)  g
  let gradients = ComputationGraph.backPropagate (fun x -> x cache) (fun x -> x cache) 1 g

  gradients |> Map.toList |> List.sortBy fst |> should equal [("Arg1", 1)]


[<Fact>]
let ``Complex back propagate`` () =
  let g = Op1 {| Id = "OpSquare"
                 Functions = Operations.Square.Functions
                 Arg =
                   Op2 {| Id = "OpMul"
                          Functions = Operations.Multiply.Functions
                          Arg0 = Arg {| Id = "OpMul-Arg0"; TrackGradient = true |}
                          Arg1 =
                            Op2 {| Id = "OpAdd"
                                   Functions = Operations.Add.Functions
                                   Arg0 = Arg {| Id = "OpAdd-Arg0"; TrackGradient = true |}
                                   Arg1 = Arg {| Id = "OpAdd-Arg1"; TrackGradient = true |} |} |} |}

  let parameters = Map.empty |> Map.add "OpMul-Arg0" 4 |> Map.add "OpAdd-Arg0" 1 |> Map.add "OpAdd-Arg1" 2
  let _, cache = ComputationGraph.forward (forwardArg parameters) g
  let gradients = ComputationGraph.backPropagate (fun x -> x cache) (fun x -> x cache) 1 g

  gradients |> Map.toList |> List.sortBy fst |> should equal [("OpAdd-Arg0", 96); ("OpAdd-Arg1", 96); ("OpMul-Arg0", 72)]
