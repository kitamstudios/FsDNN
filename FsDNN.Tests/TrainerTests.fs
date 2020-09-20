module KS.FsDNN.Tests.Trainer

open KS.FsDNN
open Xunit
open System.Collections.Generic

//[<Fact>]
let ``trainWithGD - logistic regression - OR function`` () =
  let n = Net.makeLayers 1 1.0 { N = 2 } [] (CrossEntropyLossLayer {| Classes = 1 |})

  let X = [ [ 0.; 1.; 0.; 1. ]
            [ 0.; 0.; 1.; 1. ] ] |> Tensor.ofListOfList
  let Y = [ [ 0.; 1.; 1.; 1. ] ] |> Tensor.ofListOfList

  let costs = Dictionary<int, Tensor<double>>()
  let cb = fun e _ J -> if e % 1 = 0 then costs.[e] <- J else ()
  let hp = { HyperParameters.Defaults with Epochs = 1 }

  let n = Trainer.trainWithGD cb n X Y hp

  costs.[0] |> shouldBeEquivalent [ [ 0.911103 ] ]
  n.Parameters.["W1"] |> shouldBeEquivalent [ [ -0.29620595; -0.46038118 ] ]
  n.Parameters.["b1"] |> shouldBeEquivalent [ [ 0.0034253831140481175 ] ]

[<Fact>]
let ``trainWithGD - DNN - XOR function`` () =
  ()
