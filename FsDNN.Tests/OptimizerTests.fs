namespace KS.FsDNN.Tests

open KS.FsDNN
open Xunit
open FsUnit.Xunit

module MomentumOptimizer =
  open MomentumOptimizerDomain

  [<Fact>]
  let ``initializeState test`` () =
    let W1p =
      [[ 1.62434536; -0.61175641]
       [-1.07296862;  0.86540763]] |> Tensor.ofListOfList
    let b1p =
      [1.74481176; -0.7612069] |> Tensor.ofList
    let parameters: Parameters = [("W1", W1p); ("b1", b1p)] |> Map.ofList

    let p, gv = MomentumOptimizer.initializeState parameters

    p |> Map.iter (fun k v -> v |> shouldBeEquivalentToT parameters.[k])

    gv |> Map.iter (fun k v -> v.Shape |> should equal (parameters.[k].Shape))
    gv |> Map.iter (fun _ v -> v.PointwisePower(2.).Sum() |> should equal 0.)

  [<Fact>]
  let ``updateParameters test``() =
    let W1 =
      [[ 1.62434536; -0.61175641; -0.52817175]
       [-1.07296862;  0.86540763; -2.3015387 ]] |> Tensor.ofListOfList
    let b1 =
      [1.74481176; -0.7612069] |> Tensor.ofList
    let W2 =
      [[ 0.3190391 ; -0.24937038]
       [-2.06014071; -0.3224172 ]
       [ 1.13376944; -1.09989127]] |> Tensor.ofListOfList
    let b2 =
      [-0.87785842; 0.04221375; 0.58281521] |> Tensor.ofList
    let parameters: Parameters = [("W1", W1); ("b1", b1); ("W2", W2); ("b2", b2)] |> Map.ofList

    let dW1 =
      [[-1.10061918;  1.14472371;  0.90159072]
       [ 0.50249434;  0.90085595; -0.68372786]] |> Tensor.ofListOfList
    let db1 =
      [-0.12289023; -0.93576943] |> Tensor.ofList
    let dW2 =
      [[-0.26788808;  0.53035547]
       [-0.39675353; -0.6871727 ]
       [-0.67124613; -0.0126646 ]] |> Tensor.ofListOfList
    let db2 =
      [0.2344157; 1.65980218; 0.74204416] |> Tensor.ofList
    let gradients: Gradients = [("W1", dW1); ("b1", db1); ("W2", dW2); ("b2", db2)] |> Map.ofList


    let p, gv = MomentumOptimizer.initializeState parameters

    let p, gv = MomentumOptimizer.updateParameters MomentumParameters.Defaults (TensorR0 0.01) gradients (p, gv)

    p.["W1"] |> shouldBeEquivalentTo [[1.62544598; -0.61290114; -0.52907334]; [-1.07347112; 0.86450677; -2.30085497]]
    p.["b1"] |> shouldBeEquivalentTo [[1.74493465; -0.76027113 ]]
    p.["W2"] |> shouldBeEquivalentTo [[0.31930698; -0.24990073]; [-2.05974396; -0.32173003]; [1.13444069; -1.0998786]]
    p.["b2"] |> shouldBeEquivalentTo [[-0.87809283; 0.04055394; 0.58207317 ]]

    gv.["W1"] |> shouldBeEquivalentTo [[-0.11006192; 0.11447237; 0.09015907]; [ 0.05024943; 0.09008559; -0.06837279]]
    gv.["b1"] |> shouldBeEquivalentTo [[-0.01228902; -0.09357694]]
    gv.["W2"] |> shouldBeEquivalentTo [[-0.02678881; 0.05303555]; [-0.03967535; -0.06871727]; [-0.06712461; -0.00126646]]
    gv.["b2"] |> shouldBeEquivalentTo [[0.02344157; 0.16598022; 0.07420442]]


module AdaMOptimizer =
  open AdaMOptimizerDomain

  [<Fact>]
  let ``initializeState test`` () =
    let W1p =
      [[ 1.62434536; -0.61175641]
       [-1.07296862;  0.86540763]] |> Tensor.ofListOfList
    let b1p =
      [1.74481176; -0.7612069] |> Tensor.ofList
    let parameters: Parameters = [("W1", W1p); ("b1", b1p)] |> Map.ofList

    let p, gv, sgv, t = AdaMOptimizer.initializeState parameters

    p |> Map.iter (fun k v -> v |> shouldBeEquivalentToT parameters.[k])

    gv |> Map.iter (fun k v -> v.Shape |> should equal (parameters.[k].Shape))
    gv |> Map.iter (fun _ v -> v.PointwisePower(2.).Sum() |> should equal 0.)

    sgv |> Map.iter (fun k v -> v.Shape |> should equal (parameters.[k].Shape))
    sgv |> Map.iter (fun _ v -> v.PointwisePower(2.).Sum() |> should equal 0.)

    t |> shouldBeApproximately 1.

  [<Fact>]
  let ``updateParameters test`` () =
    let W1p =
      [[ 1.62434536; -0.61175641; -0.52817175]
       [-1.07296862;  0.86540763; -2.3015387]] |> Tensor.ofListOfList
    let b1p =
      [1.74481176; -0.7612069] |> Tensor.ofList
    let W2p =
      [[ 0.3190391 ; -0.24937038]
       [-2.06014071; -0.3224172]
       [ 1.13376944; -1.09989127]] |> Tensor.ofListOfList
    let b2p =
      [-0.87785842; 0.04221375; 0.58281521] |> Tensor.ofList
    let parameters: Parameters = [("W1", W1p); ("b1", b1p); ("W2", W2p); ("b2", b2p)] |> Map.ofList

    let dW1 =
      [[-1.10061918;  1.14472371;  0.90159072]
       [ 0.50249434;  0.90085595; -0.68372786]] |> Tensor.ofListOfList
    let db1 =
      [-0.12289023; -0.93576943] |> Tensor.ofList
    let dW2 =
      [[-0.26788808;  0.53035547]
       [-0.39675353; -0.6871727 ]
       [-0.67124613; -0.0126646 ]] |> Tensor.ofListOfList
    let db2 =
      [0.2344157 ; 1.65980218; 0.74204416] |> Tensor.ofList
    let gradients: Gradients = [("W1", dW1); ("b1", db1); ("W2", dW2); ("b2", db2)] |> Map.ofList

    let p, gv, sgv, _ = AdaMOptimizer.initializeState parameters

    let p, gv, sgv, t = AdaMOptimizer.updateParameters AdaMParameters.Defaults (TensorR0 0.01) gradients (p, gv, sgv, 2.)

    p.["W1"] |> shouldBeEquivalentTo [[1.63178673; -0.61919778; -0.53561312]; [-1.08040999;  0.85796626; -2.29409733]]
    p.["b1"] |> shouldBeEquivalentTo [[1.75225313; -0.75376553]]
    p.["W2"] |> shouldBeEquivalentTo [[0.32648046; -0.25681174]; [-2.05269934; -0.31497584]; [1.14121081; -1.09244991]]
    p.["b2"] |> shouldBeEquivalentTo [[-0.88529979; 0.03477238; 0.57537385 ]]

    gv.["W1"] |> shouldBeEquivalentTo [[-0.11006192; 0.11447237; 0.09015907]; [0.05024943; 0.09008559; -0.06837279]]
    gv.["b1"] |> shouldBeEquivalentTo [[-0.01228902; -0.09357694]]
    gv.["W2"] |> shouldBeEquivalentTo [[-0.02678881; 0.05303555]; [-0.03967535; -0.06871727]; [-0.06712461; -0.00126646]]
    gv.["b2"] |> shouldBeEquivalentTo [[0.02344157; 0.16598022; 0.07420442]]

    sgv.["W1"] |> shouldBeEquivalentTo [[0.00121136; 0.00131039; 0.00081287]; [0.0002525; 0.00081154; 0.00046748]]
    sgv.["b1"] |> shouldBeEquivalentTo [[1.51020075e-05; 8.75664434e-04]]
    sgv.["W2"] |> shouldBeEquivalentTo [[7.17640232e-05; 2.81276921e-04]; [1.57413361e-04; 4.72206320e-04]; [4.50571368e-04; 1.60392066e-07]]
    sgv.["b2"] |> shouldBeEquivalentTo [[5.49507194e-05; 2.75494327e-03; 5.50629536e-04]]

    t |> shouldBeApproximately 3.
