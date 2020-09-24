module KS.FsDNN.Tests.Operations

open System
open KS.FsDNN
open Xunit

module Add =
  [<Fact>]
  let ``forwardPropagate - simple``() =
    let arg0 = [ [ 1.; 2. ] ] |> Tensor.ofListOfList
    let arg1 = [ [ 3.; 4. ] ] |> Tensor.ofListOfList

    let it = Operations.Add.Definition.Functions.F arg0 arg1

    it.[0] |> shouldBeEquivalentTo [ [ 4.; 6. ] ]
    it |> Array.tail |> shouldBeEquivalentToT2 [| arg0; arg1 |]

  [<Fact>]
  let ``backPropagate - simple``() =
    let id = "node"
    let arg0 = [ [ 1.; 2. ] ] |> Tensor.ofListOfList
    let arg1 = [ [ 3.; 4. ] ] |> Tensor.ofListOfList
    let cache = Map.empty |> Map.add id (Operations.Add.Definition.Functions.F arg0 arg1)
    let inG = [ [ 1.; 1. ] ] |> Tensor.ofListOfList

    let dArg0, dArg1 = Operations.Add.Definition.Functions.B cache id inG

    dArg0 |> shouldBeEquivalentTo [ [ 1.; 1. ] ]
    dArg1 |> shouldBeEquivalentTo [ [ 1.; 1. ] ]

module Multiply =
  [<Fact>]
  let ``forwardPropagate - simple``() =
    let arg0 = [ [ 3. ]; [ 4. ] ] |> Tensor.ofListOfList
    let arg1 = [ [ 1.; 2. ] ] |> Tensor.ofListOfList

    let it = Operations.Multiply.Definition.Functions.F arg0 arg1

    it.[0] |> shouldBeEquivalentTo [ [ 3.; 6. ]; [ 4.; 8. ] ]
    it |> Array.tail |> shouldBeEquivalentToT2 [| arg0; arg1 |]

  [<Fact>]
  let ``backPropagate - simple``() =
    let id = "node"
    let arg0 = [ [ 3. ]; [ 4. ] ] |> Tensor.ofListOfList
    let arg1 = [ [ 1.; 2. ] ] |> Tensor.ofListOfList
    let cache = Map.empty |> Map.add id (Operations.Multiply.Definition.Functions.F arg0 arg1)
    let inG =  [ [ 2.; 0. ]; [ 0.; 2. ] ] |> Tensor.ofListOfList

    let dArg0, dArg1 = Operations.Multiply.Definition.Functions.B cache id inG

    (arg0 - dArg0) |> shouldBeEquivalentTo [ [ 1. ];  [0. ] ]
    (arg1 - dArg1) |> shouldBeEquivalentTo [ [ -5.; -6. ] ]
