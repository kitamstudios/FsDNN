module KS.FsDNN.Tests.Net

open KS.FsDNN
open Xunit
open FsUnit.Xunit

[<Fact>]
let ``makeLayer - logistic regression - OR function`` () =
  let n = Net.makeLayers 1 1.0 { N = 2 } [] (L1LossLayer {| Classes = 1 |})

  n |> Net.toString |> should equal "OpL1Loss[LossLayer]( Y[TG=false], OpSigmoid[OpSigmoid]( OpAdd[OpAdd]( b1[TG=true], OpMultiply[OpMultiply]( W1[TG=true], X[TG=false] ) ) ) )"

  n.Parameters |> Map.toList |> List.map fst |> List.sort |> should equal [ "W1"; "b1" ]

  let W = [ [ -0.29934664737072975; -0.46362085300372824 ] ]
  n.Parameters.["W1"] |> shouldBeEquivalent W

  let b = [ [ 0. ] ]
  n.Parameters.["b1"] |> shouldBeEquivalent b

[<Fact>]
let ``predict - logistic regression - OR function`` () =
  let n = Net.makeLayers 1 1.0 { N = 2 } [] (L1LossLayer {| Classes = 1 |})

  let X = [ [ 0.; 1.; 0.; 1. ]
            [ 0.; 0.; 1.; 1. ] ] |> Tensor.fromListOfList
  let Y' = n |> Net.predict X

  Y' |> shouldBeEquivalent [ [ 0.5; 0.42571721; 0.38612721; 0.31800234 ] ]

[<Fact>]
let ``makeLayer - DNN`` () =
  ()

[<Fact>]
let ``predict - DNN`` () =
  ()
