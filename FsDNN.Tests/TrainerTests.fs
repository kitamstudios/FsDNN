module KS.FsDNN.Tests.Trainer

open KS.FsDNN
open Xunit
open System.Collections.Generic

(*
v multiclass
v add corresponding tests to net.predict
x remove redundant test
- Use cached value
  - in sigmoid backPropagate
  - in CCE backprop save value for softmax
- collapse sigmoid/bce and remove id layer
- Remove Tensor R0: lr should not be a tensor
- move to definitions in computation graph, make functions private

*)

[<Fact>]
let ``trainWithGD - single perceptron - logistic regression`` () =
  let n = Net.makeLayers 1 1.0 { N = 2 } [] BCEWithLogitsLossLayer

  // OR function
  let X = [ [ 0.; 1.; 0.; 1. ]
            [ 0.; 0.; 1.; 1. ] ] |> Tensor.ofListOfList
  let Y = [ [ 0.; 1.; 1.; 1. ] ] |> Tensor.ofListOfList

  let costs = Dictionary<int, Tensor<double>>()
  let cb = fun e _ J -> if e % 1 = 0 then costs.[e] <- J else ()
  let hp = { HyperParameters.Defaults with Epochs = 300; LearningRate = TensorR0 1.  }

  let n = Trainer.trainWithGD cb n X Y hp

  costs.[0] |> shouldBeEquivalentTo [ [ 0.911103 ] ]
  costs.[1] |> shouldBeEquivalentTo [ [ 0.649122 ] ]
  costs.[2] |> shouldBeEquivalentTo [ [ 0.5423996 ] ]
  n.Parameters.["W1"] |> shouldBeEquivalentTo [ [  6.17291401; 6.17174639 ] ]
  n.Parameters.["b1"] |> shouldBeEquivalentTo [ [ -2.60293958 ] ]

[<Fact>]
let ``trainWithGD - single perceptron - linear regression`` () =
  let n = Net.makeLayers 1 1.0 { N = 1 } [] MSELossLayer

  // y = 3x + 4 with randomness - refer https://repl.it/@parthopdas/linearregression#main.py
  let X = [ [8.34044009e-01; 1.44064899e+00; 2.28749635e-04; 6.04665145e-01; 2.93511782e-01; 1.84677190e-01; 3.72520423e-01; 6.91121454e-01; 7.93534948e-01; 1.07763347e+00] ] |> Tensor.ofListOfList
  let Y = [ [8.24694379; 7.56074006; 4.31972534; 5.56462506; 6.34264328; 2.49389086; 4.79514406; 5.68931001; 7.51437429; 6.13300914] ] |> Tensor.ofListOfList

  let costs = Dictionary<int, Tensor<double>>()
  let cb = fun e _ J -> if e % 1 = 0 then costs.[e] <- J else ()
  let hp = { HyperParameters.Defaults with Epochs = 300; LearningRate = TensorR0 0.1 }

  let n = Trainer.trainWithGD cb n X Y hp

  costs.[0] |> shouldBeEquivalentTo [ [ 20.3527507991 ] ]
  costs.[1] |> shouldBeEquivalentTo [ [ 15.0567184038 ] ]
  costs.[299] |> shouldBeEquivalentTo [ [ 0.6433228126 ] ]
  n.Parameters.["W1"] |> shouldBeEquivalentTo [ [ 2.83040026 ] ]
  n.Parameters.["b1"] |> shouldBeEquivalentTo [ [ 4.08554567 ] ]

[<Fact>]
let ``trainWithGD - multilayer perceptron - multilabel classification`` () =
  let n = Net.makeLayers 1 1.0 { N = 2 } [ FullyConnectedLayer {| N = 4; Activation = Sigmoid |} ] (CCEWithLogitsLossLayer {| Classes = 2 |})

  // XOR function
  let X = [ [ 0.; 0.; 1.; 1. ]
            [ 0.; 1.; 0.; 1. ] ] |> Tensor.ofListOfList
  let Y = [ [ 1.; 0.; 0.; 1. ]
            [ 0.; 1.; 1.; 0. ] ]

  let costs = Dictionary<int, Tensor<double>>()
  let cb = fun e _ J -> if e % 1 = 0 then costs.[e] <- J else ()
  let hp = { HyperParameters.Defaults with Epochs = 300; LearningRate = TensorR0 1. }

  let n = Trainer.trainWithGD cb n X (Y |> Tensor.ofListOfList) hp

  costs.[0] |> shouldBeEquivalentTo [ [ 3.44758157 ] ]
  costs.[1] |> shouldBeEquivalentTo [ [ 7.44451981 ] ]
  costs.[2] |> shouldBeEquivalentTo [ [ 8.38356992 ] ]
  costs.[299] |> shouldBeEquivalentTo [ [ 0.02101631 ] ]

  n.Parameters.["W1"] |> shouldBeEquivalentTo [ [-6.51585396;  4.70861465 ]
                                                [-3.98960860; -3.58098107 ]
                                                [-2.78070594; -2.35356575 ]
                                                [ 4.08154790; -6.17370970 ] ]
  n.Parameters.["b1"] |> shouldBeEquivalentTo [ [-2.13957179; 0.48206278; -0.65083278; -1.68797631] ]
  n.Parameters.["W2"] |> shouldBeEquivalentTo [ [-5.51024054;  2.34664527;  1.92806650; -5.90201381 ]
                                                [ 6.15139511; -3.04675126; -1.13124710;  5.73219486 ] ]
  n.Parameters.["b2"] |> shouldBeEquivalentTo [ [ 2.54121819; -2.54121819 ] ]

  let Y' = X |> Net.predict n

  Y' |> shouldBeEquivalentTo Y
