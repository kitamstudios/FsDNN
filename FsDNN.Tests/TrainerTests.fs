module KS.FsDNN.Tests.Trainer

open KS.FsDNN
open Xunit
open System.Collections.Generic

[<Fact>]
let ``trainWithGD - single perceptron - logistic regression`` () =
  let n = Net.makeLayers 1 1.0 { N = 2 } [] BinaryCrossEntropy

  // OR function
  let X = [ [ 0.; 1.; 0.; 1. ]
            [ 0.; 0.; 1.; 1. ] ] |> Tensor.ofListOfList
  let Y = [ [ 0.; 1.; 1.; 1. ] ] |> Tensor.ofListOfList

  let costs = Dictionary<int, Tensor<double>>()
  let cb = fun e _ J -> if e % 1 = 0 then costs.[e] <- J else ()
  let hp = { HyperParameters.Defaults with Epochs = 3 }

  let n = Trainer.trainWithGD cb n X Y hp

  costs.[0] |> shouldBeEquivalentTo [ [ 0.911103 ] ]
  costs.[1] |> shouldBeEquivalentTo [ [ 0.907899 ] ]
  costs.[2] |> shouldBeEquivalentTo [ [ 0.904718 ] ]
  n.Parameters.["W1"] |> shouldBeEquivalentTo [ [ -0.28995256; -0.45392965 ] ]
  n.Parameters.["b1"] |> shouldBeEquivalentTo [ [ 0.010229874835574478 ] ]

  let Y' = X |> Net.predict n

  Y' |> shouldBeEquivalentTo [[0.50255745; 0.43052177; 0.39085974; 0.32439376]]


[<Fact>]
let ``trainWithGD - single perceptron - linear regression`` () =
  let n = Net.makeLayers 1 1.0 { N = 1 } [] MeanSquaredErrorError

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

  let Y' = X |> Net.predict n

  Y' |> shouldBeEquivalentTo [ [ 6.44622405; 8.16315894; 4.08619312; 5.79699006; 4.9163015; 4.60825604; 5.13992757; 6.04169602; 6.3315672; 7.13567972] ]

[<Fact>]
let ``trainWithGD - DNN - XOR function`` () =
  ()
