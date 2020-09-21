module KS.FsDNN.Tests.Operations

open KS.FsDNN
open KS.FsDNN.Operations
open Xunit

module Add =
  [<Fact>]
  let ``forwardPropagate - simple``() =
    let arg0 = [ [ 1.; 2. ] ] |> Tensor.ofListOfList
    let arg1 = [ [ 3.; 4. ] ] |> Tensor.ofListOfList

    let it = Add.forwardPropagate arg0 arg1

    it |> shouldBeEquivalentTo [ [ 4.; 6. ] ]

  [<Fact>]
  let ``backPropagate - simple``() =
    let arg0 = [ [ 1.; 2. ] ] |> Tensor.ofListOfList
    let arg1 = [ [ 3.; 4. ] ] |> Tensor.ofListOfList
    let inG = [ [ 1.; 1. ] ] |> Tensor.ofListOfList

    let dArg0, dArg1 = Add.backPropagate inG arg0 arg1

    dArg0 |> shouldBeEquivalentTo [ [ 1.; 1. ] ]
    dArg1 |> shouldBeEquivalentTo [ [ 1.; 1. ] ]

module Multiply =
  [<Fact>]
  let ``forwardPropagate - simple``() =
    let arg0 = [ [ 3. ]; [ 4. ] ] |> Tensor.ofListOfList
    let arg1 = [ [ 1.; 2. ] ] |> Tensor.ofListOfList

    let it = Multiply.forwardPropagate arg0 arg1

    it |> shouldBeEquivalentTo [ [ 3.; 6. ]; [ 4.; 8. ] ]

  [<Fact>]
  let ``backPropagate - simple``() =
    let arg0 = [ [ 3. ]; [ 4. ] ] |> Tensor.ofListOfList
    let arg1 = [ [ 1.; 2. ] ] |> Tensor.ofListOfList
    let inG =  [ [ 2.; 0. ]; [ 0.; 2. ] ] |> Tensor.ofListOfList

    let dArg0, dArg1 = Multiply.backPropagate inG arg0 arg1

    (arg0 - dArg0) |> shouldBeEquivalentTo [ [ 1. ];  [0. ] ]
    (arg1 - dArg1) |> shouldBeEquivalentTo [ [ -5.; -6. ] ]

module BinaryCrossEntropyLoss =
  [<Fact>]
  let ``forwardPropagate - simple``() =
    let arg0 = [ [ 0. ]; [ 1. ]; [ 1. ] ] |> Tensor.ofListOfList
    let arg1 = [ [ 0.1 ]; [ 0.8 ]; [ 0.9 ] ] |> Tensor.ofListOfList

    let it = BinaryCrossEntropyLoss.forwardPropagate arg0 arg1

    it |> shouldBeEquivalentTo [ [ 0.43386458262986227 ] ]

  [<Fact>]
  let ``backPropagate - simple``() =
    let arg0 = [ [ 0. ; 1. ; 1. ] ] |> Tensor.ofListOfList
    let arg1 = [ [ 0.73105858; 0.88079708; 0.95257413 ] ] |> Tensor.ofListOfList
    let inGArray = [ [ 3.; 3.; 3. ] ]
    let inG = inGArray |> Tensor.ofListOfList

    let dArg0, dArg1 = BinaryCrossEntropyLoss.backPropagate inG arg0 arg1

    (dArg0) |> shouldBeEquivalentTo inGArray
    (arg1 - dArg1) |> shouldBeEquivalentTo [ [ -2.98722326; 2.01613236; 2.00236119 ] ]

module Sigmoid =
  [<Fact>]
  let ``forwardPropagate - simple``() =
    let arg = [ [ 1.; 2. ] ] |> Tensor.ofListOfList

    let it = Sigmoid.forwardPropagate arg

    it |> shouldBeEquivalentTo [ [ 0.73105858; 0.88079708] ]

  [<Fact>]
  let ``backPropagate - simple``() =
    let arg = [ [ 1.; 2. ] ] |> Tensor.ofListOfList
    let inG = [ [ 2.; 2. ] ] |> Tensor.ofListOfList

    let it = Sigmoid.backPropagate inG arg

    it |> shouldBeEquivalentTo [ [ 0.39322387; 0.20998717 ] ]
