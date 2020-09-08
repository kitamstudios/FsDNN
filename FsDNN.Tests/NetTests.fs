module KS.FsDNN.Tests.Net

open KS.FsDNN
open MathNet.Numerics.LinearAlgebra
open Xunit

[<Fact>]
let ``Forward propagate - 1 hidden layer`` () =
    let il = { nx = 2 }
    let ll = SoftMax {| nc = 2 |}

    let net = Net.makeLayers 0 il [] ll

    let X = matrix [[ 0.5 ]; [ 1.3 ]]

    let scores = Net.forward net X

    let expected = [[ 0.10289726 ]; [ 0.89710273 ]]
    scores |> shouldBeEquivalent expected
