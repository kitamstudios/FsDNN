namespace KS.FsDNN

open MathNet.Numerics.LinearAlgebra

module NetX =

  let _makeHiddenTransformers seed (inputLayer: InputLayer) (_: Layer list) (lossLayer: LossLayer): Transformer list =
    [ Transformer.FullyConnected.create (seed + 1) inputLayer.nx lossLayer.N ]

  let makeLayers seed (inputLayer: InputLayer) (hiddenLayers: Layer list) (lossLayer: LossLayer) =
    let its = Transformer.Input.create ()
    let hts = _makeHiddenTransformers seed inputLayer hiddenLayers lossLayer
    let lts = Transformer.Loss.create lossLayer

    { Transformers = List.concat [ [ its ]; hts; [ lts ] ] }

  let forward (net: NetX) (X: Matrix<double>): Matrix<double> =
    net.Transformers
    |> List.fold (fun acc e -> e.Forward acc) X


