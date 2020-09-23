module KS.FsDNN.Tests.Operations

open KS.FsDNN
open KS.FsDNN.Operations
open Xunit

// replace op1/2 nodes with definition
// replace test with definitions
// one test for name sanity using reflection

module Add =
  [<Fact>]
  let ``forwardPropagate - simple``() =
    let arg0 = [ [ 1.; 2. ] ] |> Tensor.ofListOfList
    let arg1 = [ [ 3.; 4. ] ] |> Tensor.ofListOfList

    let it = Add.forwardPropagate arg0 arg1

    it |> shouldBeEquivalentTo [ [ 4.; 6. ] ]

  [<Fact>]
  let ``backPropagate - simple``() =
    let id = "node"
    let arg0 = [ [ 1.; 2. ] ] |> Tensor.ofListOfList
    let arg1 = [ [ 3.; 4. ] ] |> Tensor.ofListOfList
    let cache = Map.empty |> Map.add id [| arg0; arg1 |]
    let inG = [ [ 1.; 1. ] ] |> Tensor.ofListOfList

    let dArg0, dArg1 = Add.backPropagate cache id inG

    dArg0 |> shouldBeEquivalentTo [ [ 1.; 1. ] ]
    dArg1 |> shouldBeEquivalentTo [ [ 1.; 1. ] ]

module Multiply =
  [<Fact>]
  let ``forwardPropagate - simple``() =
    let arg0 = [ [ 3. ]; [ 4. ] ] |> Tensor.ofListOfList
    let arg1 = [ [ 1.; 2. ] ] |> Tensor.ofListOfList

    let it = Multiply.forwardPropagate arg0 arg1

    it |> shouldBeEquivalentTo [ [ 3.; 6. ]; [ 4.; 8. ] ]

  [<Fact>]
  let ``backPropagate - simple``() =
    let id = "node"
    let arg0 = [ [ 3. ]; [ 4. ] ] |> Tensor.ofListOfList
    let arg1 = [ [ 1.; 2. ] ] |> Tensor.ofListOfList
    let cache = Map.empty |> Map.add id [| arg0; arg1 |]
    let inG =  [ [ 2.; 0. ]; [ 0.; 2. ] ] |> Tensor.ofListOfList

    let dArg0, dArg1 = Multiply.backPropagate cache id inG

    (arg0 - dArg0) |> shouldBeEquivalentTo [ [ 1. ];  [0. ] ]
    (arg1 - dArg1) |> shouldBeEquivalentTo [ [ -5.; -6. ] ]

module BCEWithLogitsLoss =
  [<Fact>]
  let ``forwardPropagate - simple``() =
    let arg0 = [ [ 0. ]; [ 1. ]; [ 1. ] ] |> Tensor.ofListOfList
    let arg1 = [ [ 0.1 ]; [ 0.8 ]; [ 0.9 ] ] |> Tensor.ofListOfList

    let it = BCEWithLogitsLoss.forwardPropagate arg0 arg1

    it |> shouldBeEquivalentTo [ [ 0.43386458262986227 ] ]

  [<Fact>]
  let ``backPropagate - simple``() =
    let id = "node"
    let arg0 = [ [ 0. ; 1. ; 1. ] ] |> Tensor.ofListOfList
    let arg1 = [ [ 0.73105858; 0.88079708; 0.95257413 ] ] |> Tensor.ofListOfList
    let cache = Map.empty |> Map.add id [| arg0; arg1 |]
    let inGArray = [ [ 3.; 3.; 3. ] ]
    let inG = inGArray |> Tensor.ofListOfList

    let dArg0, dArg1 = BCEWithLogitsLoss.backPropagate cache id inG

    (dArg0) |> shouldBeEquivalentTo inGArray
    (arg1 - dArg1) |> shouldBeEquivalentTo [ [ -2.98722326; 2.01613236; 2.00236119 ] ]

module CCEWithLogitsLoss =
  [<Fact>]
  let ``forwardPropagate - simple``() =
    let arg0 = [ [1.; 0.]
                 [0.; 1.]
                 [0.; 1.]
                 [1.; 0.] ] |> Tensor.ofListOfList
    let arg1 = [ [  0.6693461;  0.3306539 ]
                 [  0.6858448;  0.3141552 ]
                 [  0.67993372; 0.32006628]
                 [  0.69203662; 0.30796338] ] |> Tensor.ofListOfList

    let it = CCEWithLogitsLoss.Definition.Functions.F arg0 arg1

    it |> shouldBeEquivalentTo [ [ 3.06666578295317 ] ]

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

    let dArg0, dArg1 = CCEWithLogitsLoss.backPropagate cache id inG

    let gradient =  [ [ -1.459516065;  1.459516065]
                      [ 0.033249825;  -0.033249825]
                      [ 0.02562897;  -0.02562897]
                      [ -1.47777225;   1.47777225 ] ]

    dArg0 |> shouldBeEquivalentTo [ [ 1.5 ] ]
    dArg1 |> shouldBeEquivalentTo gradient
    (arg1 - dArg1) |> shouldBeEquivalentTo [ [ 1.48650535; -0.48650535 ]
                                             [ -0.01108327; 1.01108327 ]
                                             [ -0.00854299; 1.00854299 ]
                                             [ 1.49259075; -0.49259075 ] ]

module MSELoss =
  [<Fact>]
  let ``forwardPropagate - simple``() =
    let arg0 = [ [ 1.; 2.; 3. ] ] |> Tensor.ofListOfList
    let arg1 = [ [ 4.; 5.; 6. ] ] |> Tensor.ofListOfList

    let it = MSELoss.forwardPropagate arg0 arg1

    it |> shouldBeEquivalentTo [ [ 4.5 ] ]

  [<Fact>]
  let ``backPropagate - simple``() =
    let id = "node"
    let arg0 = [ [ 1.; 2.; 3. ] ] |> Tensor.ofListOfList
    let arg1 = [ [ 4.; 5.; 6. ] ] |> Tensor.ofListOfList
    let cache = Map.empty |> Map.add id [| arg0; arg1 |]
    let inGArray = [ [ 2.; 2.; 2. ] ]
    let inG = inGArray |> Tensor.ofListOfList

    let dArg0, dArg1 = MSELoss.backPropagate cache id inG

    (dArg0) |> shouldBeEquivalentTo inGArray
    (arg1 - dArg1) |> shouldBeEquivalentTo [ [ 2.; 3.; 4. ] ]

module Sigmoid =
  [<Fact>]
  let ``forwardPropagate - simple``() =
    let arg = [ [ 1.; 2. ] ] |> Tensor.ofListOfList

    let it = Sigmoid.forwardPropagate arg

    it |> shouldBeEquivalentTo [ [ 0.73105858; 0.88079708] ]

  [<Fact>]
  let ``backPropagate - simple``() =
    let id = "node"
    let arg = [ [ 1.; 2. ] ] |> Tensor.ofListOfList
    let cache = Map.empty |> Map.add id [| arg |]
    let inG = [ [ 2.; 2. ] ] |> Tensor.ofListOfList

    let it = Sigmoid.backPropagate cache id inG

    it |> shouldBeEquivalentTo [ [ 0.39322387; 0.20998717 ] ]
