module KS.FsDNN.Tests.ComputationGraph

open KS.FsDNN
open Xunit
open FsUnit.Xunit

type Operations1 =
  | OpSquare

type Operations2 =
  | OpAdd
  | OpMul

let evaluate (g: ComputationGraph<int, Operations1, Operations2>): int =

  let fArg = id

  let fOperation1 (o: Operations1) arg =
    match o with
    | OpSquare -> arg * arg

  let fOperation (o: Operations2) arg0 arg1 =
    match o with
    | OpAdd -> arg0 + arg1
    | OpMul -> arg0 * arg1

  ComputationGraph.foldBack fArg fOperation1 fOperation g

[<Fact>]
let ``Simple computation`` () =

  let g = Operation2 {| Op = OpAdd; Arg0 = Arg 1; Arg1 = 2; |}

  let x = evaluate g

  x |> should equal 3

[<Fact>]
let ``Complex computation`` () =

  let g = Operation1 {| Op = OpSquare
                        Arg =
                          Operation2 {| Op = OpMul
                                        Arg0 =
                                          Operation2 {| Op = OpAdd
                                                        Arg0 = Arg 2
                                                        Arg1 = 1 |}
                                        Arg1 = 4 |} |}

  let x = evaluate g

  x |> should equal 144
