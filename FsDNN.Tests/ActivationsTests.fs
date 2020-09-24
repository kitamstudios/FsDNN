module KS.FsDNN.Tests.Activations

open KS.FsDNN
open Xunit

module Sigmoid =
  [<Fact>]
  let ``forwardPropagate - simple``() =
    let arg = [ [ 1.; 2. ] ] |> Tensor.ofListOfList

    let it = Activations.Sigmoid.Definition.Functions.F arg

    it.[0] |> shouldBeEquivalentTo [ [ 0.73105858; 0.88079708] ]
    it |> Array.tail |> shouldBeEquivalentToT2 [| arg |]

  [<Fact>]
  let ``backPropagate - simple``() =
    let id = "node"
    let arg = [ [ 1.; 2. ] ] |> Tensor.ofListOfList
    let cache = Map.empty |> Map.add id (Activations.Sigmoid.Definition.Functions.F arg)
    let inG = [ [ 2.; 2. ] ] |> Tensor.ofListOfList

    let it = Activations.Sigmoid.Definition.Functions.B cache id inG

    it |> shouldBeEquivalentTo [ [ 0.39322387; 0.20998717 ] ]

module SoftMax =
  [<Fact>]
  let ``forwardPropagate - simple``() =
    let arg = [ [  1.; -4. ]
                [ -1.; -3. ] ] |> Tensor.ofListOfList

    let it = Activations.SoftMax.Definition.Functions.F arg

    it.[0] |> shouldBeEquivalentTo [ [ 0.88079707; 0.26894142 ]
                                     [ 0.11920292; 0.73105857 ] ]
    it |> Array.tail |> shouldBeEquivalentToT2 [| arg |]

module HardMax =
  [<Fact>]
  let ``forwardPropagate - simple``() =
    let arg = [ [  1.; -4. ]
                [ -1.; -3. ] ] |> Tensor.ofListOfList

    let it = Activations.HardMax.Definition.Functions.F arg

    it.[0] |> shouldBeEquivalentTo [ [ 1.; 0. ]
                                     [ 0.; 1. ] ]
    it |> Array.tail |> shouldBeEquivalentToT2 [| arg |]

module ReLU =
  [<Fact>]
  let ``forwardPropagate - simple``() =
    let arg = [ [ -1.; 2. ] ] |> Tensor.ofListOfList

    let it = Activations.ReLU.Definition.Functions.F arg

    it.[0] |> shouldBeEquivalentTo [ [ 0.; 2. ] ]
    it |> Array.tail |> shouldBeEquivalentToT2 [| arg |]

  [<Fact>]
  let ``backPropagate - simple``() =
    let id = "node"
    let arg = [ [ -1.; 2. ] ] |> Tensor.ofListOfList
    let cache = Map.empty |> Map.add id (Activations.ReLU.Definition.Functions.F arg)
    let inG = [ [ 2.; 2. ] ] |> Tensor.ofListOfList

    let it = Activations.ReLU.Definition.Functions.B cache id inG

    it |> shouldBeEquivalentTo [ [ 0.; 2. ] ]
