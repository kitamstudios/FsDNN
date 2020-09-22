namespace KS.FsDNN

(*

  # Notation & Dimensional Analysis

  ## Formula

  m  = # of training input feature vectors
  nₓ = Dimension of input feature vector
  X = Input
  A₀ = X
  Zₗ = Wₗ.Aₗ₋₁ + bₗ
  Aₗ = gₗ(Zₗ)
  Ŷ = Output of the last layer
  Y = Expected output
  L = # layers

  dim(X) = nₓ x m
  dim(Wₗ) = nₗ x nₗ₋₁
  dim(bₗ) = nₗ x 1
  dim(Zₗ) = nₗ₋₁ x m
  dim(Aₗ) = dim(Zₗ)

  ## Example

  nₓ      = 3
  Layers  = 4, 5, 2
  m       = 7
  L       = 3

  [  W1    A0    b1      A1   ]    [  W2    A1    b2      A2   ]    [  W3    A2    b3   =  A3   ]
  [ (4x3).(3x7)+(4x1) = (4x7) ] -> [ (5x4).(4x7)+(5x1) = (5x7) ] -> [ (2x5).(5x7)+(2x1) = (2x7) ]

 *)

[<AutoOpen>]
module NetDomain =

  type InputLayer =
    { N: int }

  type HiddenLayer =
    | FullyConnectedLayer of {| N: int |}

    member this.N =
      match this with
      | FullyConnectedLayer l -> l.N

  type LossLayer =
    | BinaryCrossEntropy
    | MeanSquaredErrorError

    member this.Classes =
      match this with
      | BinaryCrossEntropy -> 1
      | MeanSquaredErrorError -> 1

  type ComputationGraph = ComputationGraph<Tensor<double>>

  type Parameters = Map<string, Tensor<double>>

  type Net =
    { LossGraph: ComputationGraph
      PredictGraph: ComputationGraph
      Parameters: Parameters }

module Net =

  let toString n: string =
    n.LossGraph |> ComputationGraph.toString

  let private _initializeParameters (seed: int) (heScale: double) layerNeurons: Map<string, Tensor<double>> =
    let ws =
      layerNeurons
      |> List.pairwise
      |> List.mapi (fun i (nPrev, n) -> sprintf "W%d" (i + 1), Tensor.createRandomizedR2 (seed + i) n nPrev heScale)

    let bs =
      layerNeurons
      |> List.skip 1
      |> List.mapi (fun i n -> sprintf "b%d" (i + 1), (Tensor.createZerosR1 n))

    ws @ bs
    |> Map.ofList

  let private _createComputationGraphForOneLayer prevLayerCG id: ComputationGraph =
     Op2 {| Id = Operations.Add.Definition.Name
            Functions = Operations.Add.Definition.Functions
            Arg0 =
              Op2 {| Id = Operations.Multiply.Definition.Name
                     Functions = Operations.Multiply.Definition.Functions
                     Arg0 = Arg {| Id = sprintf "W%d" id; TrackGradient = true |}
                     Arg1 = prevLayerCG |}
            Arg1 = Arg {| Id = sprintf "b%d" id; TrackGradient = true |} |}

  let _makeImplicitHiddenLayerForLossLayer g lossLayer =
    let fns =
      match lossLayer with
      | BinaryCrossEntropy -> Operations.Sigmoid.Definition.Functions
      | MeanSquaredErrorError -> Operations.Linear.Functions

    Op1 {| Id = Operations.Sigmoid.Definition.Name
           Functions = fns
           Arg = g |}

  let private _createComputationGraphForPrediction (hiddenLayers: HiddenLayer list) (lossLayer: LossLayer): ComputationGraph =
    let g0 = Arg {| Id = "X"; TrackGradient = false |}

    let g =
      (hiddenLayers |> List.map (fun l -> l.N)) @ [lossLayer.Classes]
      |> List.fold (fun (g, id) _ -> _createComputationGraphForOneLayer g id, id + 1) (g0, 1)
      |> fst

    _makeImplicitHiddenLayerForLossLayer g lossLayer

  let private _makeLossLayer lossLayer g =
    let fns =
      match lossLayer with
      | BinaryCrossEntropy -> Operations.BinaryCrossEntropyLoss.Definition
      | MeanSquaredErrorError _ -> Operations.MeanSquaredErrorLoss.Definition

    let lg = Op2 {| Id = fns.Name
                    Functions = fns.Functions
                    Arg0 = Arg {| Id = "Y"; TrackGradient = false |}
                    Arg1 = g |}

    lg

  let makeLayers seed heScale (inputLayer: InputLayer) (hiddenLayers: HiddenLayer list) (lossLayer: LossLayer): Net =
    let pg = _createComputationGraphForPrediction hiddenLayers lossLayer

    let lg = _makeLossLayer lossLayer pg

    let ps = _initializeParameters seed heScale [ inputLayer.N; lossLayer.Classes ]

    { LossGraph = lg
      PredictGraph = pg
      Parameters = ps }

  let private _forwardArg parameters id: Tensor<double> =
    parameters |> Map.find id

  let predict n (X: Tensor<double>) =
    let parameters = n.Parameters |> Map.add "X" X
    ComputationGraph.predict (_forwardArg parameters) n.PredictGraph

  let forwardPropagate n (X: Tensor<double>) (Y: Tensor<double>) =
    let parameters = n.Parameters |> Map.add "X" X |> Map.add "Y" Y
    ComputationGraph.forward (_forwardArg parameters) n.LossGraph

  let backPropagate n cache =
    let inline f x = x cache
    ComputationGraph.backPropagate f f Scalar1 n.LossGraph
