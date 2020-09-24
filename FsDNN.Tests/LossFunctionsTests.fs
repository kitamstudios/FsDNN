module KS.FsDNN.Tests.LossFunctions

open KS.FsDNN
open Xunit

module BCEWithLogitsLoss =
  [<Fact>]
  let ``forwardPropagate - simple``() =
    let arg0 = [ [ 0. ]; [ 1. ]; [ 1. ] ] |> Tensor.ofListOfList
    let arg1 = [ [ 0.1 ]; [ 0.8 ]; [ 0.9 ] ] |> Tensor.ofListOfList

    let it = LossFunctions.BCEWithLogitsLoss.Definition.Functions.F arg0 arg1

    it |> shouldBeEquivalentTo [ [ 0.43386458262986227 ] ]

  [<Fact>]
  let ``backPropagate - simple``() =
    let id = "node"
    let arg0 = [ [ 0. ; 1. ; 1. ] ] |> Tensor.ofListOfList
    let arg1 = [ [ 0.73105858; 0.88079708; 0.95257413 ] ] |> Tensor.ofListOfList
    let cache = Map.empty |> Map.add id [| arg0; arg1 |]
    let inGArray = [ [ 3.; 3.; 3. ] ]
    let inG = inGArray |> Tensor.ofListOfList

    let dArg0, dArg1 = LossFunctions.BCEWithLogitsLoss.Definition.Functions.B cache id inG

    (dArg0) |> shouldBeEquivalentTo inGArray
    (arg1 - dArg1) |> shouldBeEquivalentTo [ [ -2.98722326; 2.01613236; 2.00236119 ] ]

module CCEWithLogitsLoss =
  [<Fact>]
  let ``forwardPropagate - simple``() =
    let arg0 = [ [0.; 0.]
                 [0.; 1.]
                 [1.; 0.]
                 [1.; 1.] ] |> Tensor.ofListOfList
    let arg1 = [ [ 0.6693461;  -0.3306539  ]
                 [ 0.6858448;   0.3141552  ]
                 [-0.67993372;  0.32006628 ]
                 [ 0.69203662;  0.30796338 ] ] |> Tensor.ofListOfList

    let it = LossFunctions.CCEWithLogitsLoss.Definition.Functions.F arg0 arg1

    it |> shouldBeEquivalentTo [ [ 6.23951519 ] ]

  [<Fact>]
  let ``backPropagate - simple``() =
    let id = "node"
    let arg0 = [ [ 1.; 0.]
                 [ 0.; 1.]
                 [ 0.; 1.]
                 [ 1.; 0.] ] |> Tensor.ofListOfList
    let arg1 = [ [ 0.02698929; 0.97301071]
                 [ 0.02216655; 0.97783345]
                 [ 0.01708598; 0.98291402]
                 [ 0.0148185;  0.9851815 ] ] |> Tensor.ofListOfList

    let cache = Map.empty |> Map.add id [| arg0; arg1 |]
    let inG = 1.5 |> TensorR0

    let dArg0, dArg1 = LossFunctions.CCEWithLogitsLoss.Definition.Functions.B cache id inG

    let gradient =  [ [ -1.12247410;  0.37248275 ]
                      [  0.37570956; -1.12571651 ]
                      [  0.37380558; -1.12381010 ]
                      [ -1.12704104;  0.37704386 ] ]

    dArg0 |> shouldBeEquivalentTo [ [ 1.5 ] ]
    dArg1 |> shouldBeEquivalentTo gradient
    (arg1 - dArg1) |> shouldBeEquivalentTo [ [  1.14946339; 0.60052795 ]
                                             [ -0.35354301; 2.10354996 ]
                                             [ -0.35671960; 2.10672412 ]
                                             [  1.14185954; 0.60813763 ] ]

module MSELoss =
  [<Fact>]
  let ``forwardPropagate - simple``() =
    let arg0 = [ [ 1.; 2.; 3. ] ] |> Tensor.ofListOfList
    let arg1 = [ [ 4.; 5.; 6. ] ] |> Tensor.ofListOfList

    let it = LossFunctions.MSELoss.Definition.Functions.F arg0 arg1

    it |> shouldBeEquivalentTo [ [ 4.5 ] ]

  [<Fact>]
  let ``backPropagate - simple``() =
    let id = "node"
    let arg0 = [ [ 1.; 2.; 3. ] ] |> Tensor.ofListOfList
    let arg1 = [ [ 4.; 5.; 6. ] ] |> Tensor.ofListOfList
    let cache = Map.empty |> Map.add id [| arg0; arg1 |]
    let inGArray = [ [ 2.; 2.; 2. ] ]
    let inG = inGArray |> Tensor.ofListOfList

    let dArg0, dArg1 = LossFunctions.MSELoss.Definition.Functions.B cache id inG

    (dArg0) |> shouldBeEquivalentTo inGArray
    (arg1 - dArg1) |> shouldBeEquivalentTo [ [ 2.; 3.; 4. ] ]
