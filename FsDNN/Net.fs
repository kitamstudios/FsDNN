namespace KS.FsDNN

open System
open MathNet.Numerics.Distributions
open MathNet.Numerics.LinearAlgebra

module Net =

  let _makeHiddenTransformers seed (inputLayer: InputLayer) (hiddenLayers: Layer list) (lossLayer: LossLayer): Transformer list =
    let n = match lossLayer with | SoftMax svm -> svm.nc

    let W = DenseMatrix.random<double> n inputLayer.nx (Normal(0., 1., Random(seed + 1)))
    let b = DenseMatrix.create n 1 0.

    [ { n = n; W = W; b = b; forward = fun a -> W * a + b } ]

  let _makeLossTransformer (lossLayer: LossLayer): Transformer =
    let n = match lossLayer with | SoftMax svm -> svm.nc
    let f (x: Matrix<double>) =
      let expx = x.PointwiseExp();
      expx.Divide(expx.ColumnSums().Sum() + Constants.DivideBy0Guard)
    { n = n; W = null; b = null; forward = f }

  let makeLayers seed (inputLayer: InputLayer) (hiddenLayers: Layer list) (lossLayer: LossLayer) =
    let its = { n = inputLayer.nx; W = null; b = null; forward = id }
    let hts = _makeHiddenTransformers seed inputLayer hiddenLayers lossLayer
    let lts = _makeLossTransformer lossLayer

    { Transformers = List.concat [ [ its ]; hts; [ lts ] ] }

  let forward (net: Net) (X: Matrix<double>): Matrix<double> =
    net.Transformers
    |> List.fold (fun acc e -> e.forward acc) X
