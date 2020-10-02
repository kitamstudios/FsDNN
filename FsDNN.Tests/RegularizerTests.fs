namespace KS.FsDNN.Tests

open KS.FsDNN
open Xunit

module L2Regularizer =

  [<Fact>]
  let ``regularizeCost test`` () =
    let W1 = [ [ 0.1; 0.2 ]; [ 0.3; 0.4; ]; [ 0.5; 0.6] ] |> Tensor.ofListOfList
    let b1 = [ 9.1 ] |> Tensor.ofList
    let W2 = [ [ 0.9; 0.7; 0.33 ] ] |> Tensor.ofListOfList
    let parameters = Map.empty |> Map.add "W1" W1 |> Map.add "W2" W2 |> Map.add "b1" b1
    let cost = 2.718 |> TensorR0
    let cost' = L2Regularizer.regularizeCost 0.15 19. parameters cost
    cost' |> shouldBeEquivalentTo [ [ 2.72715355; ] ]

  [<Fact>]
  let ``regularizeGradients test``() =
    let W1 = [ [ 0.1; 0.2 ]; [ 0.3; 0.4; ]; [ 0.5; 0.6] ] |> Tensor.ofListOfList
    let b1 = [ 9.1 ] |> Tensor.ofList
    let W2 = [ [ 0.9; 0.7; 0.33 ] ] |> Tensor.ofListOfList
    let parameters = Map.empty |> Map.add "W1" W1 |> Map.add "W2" W2 |> Map.add "b1" b1

    let dW1 = [ [ -0.1; 0.12 ]; [ 0.13; -0.4; ]; [ -0.15; -0.6] ] |> Tensor.ofListOfList
    let db1 = [ 119.1 ] |> Tensor.ofList
    let dW2 = [ [ -10.9; 0.17; 7.33 ] ] |> Tensor.ofListOfList
    let gradients = Map.empty |> Map.add "W1" dW1 |> Map.add "W2" dW2 |> Map.add "b1" db1

    let gradients' = L2Regularizer.regularizeGradients 0.15 19. parameters gradients

    gradients'.["W1"] |> shouldBeEquivalentTo [ [ -0.09921052; 0.121578947;]; [ 0.1323684210; -0.396842105;]; [ -0.146052631; -0.595263157;] ]
    gradients'.["b1"] |> shouldBeEquivalentToT db1
    gradients'.["W2"] |> shouldBeEquivalentTo [ [ -10.892894736842106; 0.1755263157894737; 7.332605263157895 ]  ]
