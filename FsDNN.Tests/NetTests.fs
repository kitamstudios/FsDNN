module KS.FsDNN.Tests.Net

open KS.FsDNN
open Xunit
open FsUnit.Xunit

[<Fact>]
let ``makeLayer - logistic unit`` () =
  let n = Net.makeLayers 1 1.0 { N = 2 } [] (BCEWithLogitsLossLayer {| Classes = 1 |})

  let it = n |> Net.toString
  it.LossGraphString |> should equal "BCEWithLogitsLoss[Loss,1]( Y[TG=false], Add1( Multiply1( W1[TG=true], X[TG=false] ), b1[TG=true] ) )"
  it.PredictGraphString |> should equal "Sigmoid[Predict,1]( Add1( Multiply1( W1[TG=true], X[TG=false] ), b1[TG=true] ) )"

  n.Parameters |> Map.toList |> List.map fst |> List.sort |> should equal [ "W1"; "b1" ]
  n.Parameters.["W1"] |> shouldBeEquivalentTo [ [ -0.29934664737072975; -0.46362085300372824 ] ]
  n.Parameters.["b1"] |> shouldBeEquivalentTo [ [ 0. ] ]

[<Fact>]
let ``makeLayer - multilayer perceptron`` () =
  let n = Net.makeLayers 1 1.0 { N = 2 } [ FullyConnectedLayer {| N = 3; Activation = Sigmoid |} ] (CCEWithLogitsLossLayer {| Classes = 2 |})

  let it = n |> Net.toString
  it.LossGraphString |> should equal "CCEWithLogitsLoss[Loss,2]( Y[TG=false], Add2( Multiply2( W2[TG=true], Sigmoid( Add1( Multiply1( W1[TG=true], X[TG=false] ), b1[TG=true] ) ) ), b2[TG=true] ) )"
  it.PredictGraphString |> should equal "HardMax[Predict,2]( Add2( Multiply2( W2[TG=true], Sigmoid( Add1( Multiply1( W1[TG=true], X[TG=false] ), b1[TG=true] ) ) ), b2[TG=true] ) )"

  n.Parameters |> Map.toList |> List.map fst |> List.sort |> should equal [ "W1"; "W2"; "b1"; "b2" ]
  n.Parameters.["W1"].Shape |> should equal (3, 2)
  n.Parameters.["b1"].Shape |> should equal (3, 1)
  n.Parameters.["W2"].Shape |> should equal (2, 3)
  n.Parameters.["b2"].Shape |> should equal (2, 1)


[<Fact>]
let ``predict - single perceptron - logistic regression`` () =
  let n = Net.makeLayers 1 1.0 { N = 2 } [] (BCEWithLogitsLossLayer {| Classes = 1 |})
  let parameters =
    Map.empty
    |> Map.add "W1" ([ [  6.17291401; 6.17174639 ] ] |> Tensor.ofListOfList)
    |> Map.add "b1" ([ -2.60293958 ] |> Tensor.ofList)
  let n = { n with Parameters = parameters }

  // OR function
  let X = [ [ 0.; 1.; 0.; 1. ]
            [ 0.; 0.; 1.; 1. ] ] |> Tensor.ofListOfList

  let Y' = X |> Net.predict n

  Y' |> shouldBeEquivalentTo [[ 0.06894947; 0.97261450; 0.97258339; 0.99994122 ]] // Nearly equal to [ 0.; 1.; 1.; 1. ]

[<Fact>]
let ``predict - single perceptron - linear regression`` () =
  let n = Net.makeLayers 1 1.0 { N = 1 } [] MSELossLayer
  let parameters =
    Map.empty
    |> Map.add "W1" ([ [ 2.83040026 ] ] |> Tensor.ofListOfList)
    |> Map.add "b1" ([ 4.08554567 ] |> Tensor.ofList)
  let n = { n with Parameters = parameters }

  // y = 3x + 4 with randomness - refer https://repl.it/@parthopdas/linearregression#main.py
  let X = [ [8.34044009e-01; 1.44064899e+00; 2.28749635e-04; 6.04665145e-01; 2.93511782e-01; 1.84677190e-01; 3.72520423e-01; 6.91121454e-01; 7.93534948e-01; 1.07763347e+00] ] |> Tensor.ofListOfList

  let Y' = X |> Net.predict n
  Y' |> shouldBeEquivalentTo [ [ 6.44622405; 8.16315894; 4.08619312; 5.79699006; 4.9163015; 4.60825604; 5.13992757; 6.04169602; 6.3315672; 7.13567972] ]

[<Fact>]
let ``predict - multilayer perceptron - multilabel classification`` () =
  let n = Net.makeLayers 1 1.0 { N = 2 } [ FullyConnectedLayer {| N = 4; Activation = Sigmoid |} ] (CCEWithLogitsLossLayer {| Classes = 2 |})
  let parameters =
    Map.empty
    |> Map.add "W1" ([ [-6.51585396;  4.70861465 ]; [-3.98960860; -3.58098107 ];  [-2.78070594; -2.35356575 ];  [ 4.08154790; -6.17370970 ] ] |> Tensor.ofListOfList)
    |> Map.add "b1" ([ -2.13957179; 0.48206278; -0.65083278; -1.68797631 ] |> Tensor.ofList)
    |> Map.add "W2" ([ [-5.51024054;  2.34664527;  1.92806650; -5.90201381 ]; [ 6.15139511; -3.04675126; -1.13124710;  5.73219486 ] ] |> Tensor.ofListOfList)
    |> Map.add "b2" ([ 2.54121819; -2.54121819 ] |> Tensor.ofList)
  let n = { n with Parameters = parameters }

  // XOR function
  let X = [ [ 0.; 0.; 1.; 1. ]
            [ 0.; 1.; 0.; 1. ] ] |> Tensor.ofListOfList

  let Y' = X |> Net.predict n

  let Y = [ [ 1.; 0.; 0.; 1. ]
            [ 0.; 1.; 1.; 0. ] ]
  Y' |> shouldBeEquivalentTo Y
