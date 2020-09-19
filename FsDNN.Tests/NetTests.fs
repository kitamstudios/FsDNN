module KS.FsDNN.Tests.Net

open KS.FsDNN
open Xunit
open FsUnit.Xunit

[<Fact>]
let ``makeLayer - logistic regression`` () =
  let n = Net.makeLayers 1 1.0 { N = 3 } [] (SoftMax {| Classes = 2 |})

  n |> Net.toString |> should equal "OpL1Loss[L1Loss]( Y[TG=false], OpSigmoid[OpSigmoid]( OpAdd[OpAdd]( b[TG=true], OpMultiply[OpMultiply]( W[TG=true], X[TG=false] ) ) ) )"

  n.Parameters |> Map.toList |> List.map fst |> List.sort |> should equal [ "W"; "b" ]

  let W = [ [ -0.24441551409037851; -0.37854484132434002; -0.15288972607056753 ] ]
  n.Parameters.["W"] |> shouldBeEquivalent W

  let b = [ [ 0. ] ]
  n.Parameters.["b"] |> shouldBeEquivalent b

[<Fact>]
let ``predict - logistic regression`` () =
  let n = Net.makeLayers 1 1.0 { N = 3 } [] (SoftMax {| Classes = 2 |})

  let X = [ [ 750. ]; [ 3.9 ]; [ 4. ] ] |> toM
  let Y' = n |> Net.predict X

  Y' |> shouldBeEquivalent [ [ 3.03397359934e-81 ] ]

[<Fact>]
let ``makeLayer - DNN`` () =
  ()

[<Fact>]
let ``predict - DNN`` () =
  ()
