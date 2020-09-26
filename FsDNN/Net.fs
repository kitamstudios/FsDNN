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

  type Activation =
    | Linear
    | Sigmoid
    | ReLU

  type HiddenLayer =
    | FullyConnectedLayer of {| N: int; Activation: Activation |}

    member this.N =
      match this with
      | FullyConnectedLayer l -> l.N

    member this.Activation =
      match this with
      | FullyConnectedLayer l -> l.Activation

  /// Following has the exhaustive list
  /// https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/
  type LossLayer =
    | BCEWithLogitsLossLayer of {| Classes: int  |}
    | CCEWithLogitsLossLayer of {| Classes: int  |}
    | MSELossLayer

    member this.Classes =
      match this with
      | BCEWithLogitsLossLayer l -> l.Classes
      | CCEWithLogitsLossLayer l -> l.Classes
      | MSELossLayer -> 1

  type ComputationGraph = ComputationGraph<Tensor<double>>

  type Parameters = Map<string, Tensor<double>>

  type Gradients = Map<string, Tensor<double>>

  /// Exponentially weighted average of the gradients.
  type GradientVelocities = Map<string, Tensor<double>>

  /// Exponentially weighted average of the squared gradient.
  type SquaredGradientVelocities = Map<string, Tensor<double>>

  type TensorInitializer = int -> int -> int -> Tensor<double>

  type Net =
    { LossGraph: ComputationGraph
      PredictGraph: ComputationGraph
      Parameters: Parameters }

module Net =

  let toString n =
    {| LossGraphString = n.LossGraph |> ComputationGraph.toString
       PredictGraphString = n.PredictGraph |> ComputationGraph.toString |}

  let initializeParameters seed0 (initTensors: TensorInitializer) layerSizes: Map<string, Tensor<double>> =

    let ws =
      layerSizes
      |> List.pairwise
      |> List.mapi (fun i (nPrev, n) -> sprintf "W%d" (i + 1), initTensors (seed0 + i) n nPrev)

    let bs =
      layerSizes
      |> List.skip 1
      |> List.mapi (fun i n -> sprintf "b%d" (i + 1), (Tensor.createZerosR1 n))

    ws @ bs
    |> Map.ofList

  let private _createComputationGraphForHiddenLayer prevLayerCG a id: ComputationGraph =
    let linear =
       Op2 {| D = { Operations.Add.Definition with Name = sprintf "%s%d" Operations.Add.Definition.Name id }
              Arg0 =
                Op2 {| D = { Operations.Multiply.Definition with Name = sprintf "%s%d" Operations.Multiply.Definition.Name id }
                       Arg0 = Arg {| Id = sprintf "W%d" id; TrackGradient = true |}
                       Arg1 = prevLayerCG |}
              Arg1 = Arg {| Id = sprintf "b%d" id; TrackGradient = true |} |}

    let d =
      match a with
      | Linear -> Activations.Linear.Definition
      | Sigmoid -> Activations.Sigmoid.Definition
      | ReLU -> Activations.ReLU.Definition

    Op1 {| D = d
           Arg = linear |}

  let private _createCommonComputationGraph (hiddenLayers: HiddenLayer list): ComputationGraph =
    let g0 = Arg {| Id = "X"; TrackGradient = false |}

    let hg, id =
      hiddenLayers
      |> List.fold (fun (g, id) l -> _createComputationGraphForHiddenLayer g l.Activation id, id + 1) (g0, 1)

    let g =
       Op2 {| D = { Operations.Add.Definition with Name = sprintf "%s%d" Operations.Add.Definition.Name id }
              Arg0 =
                Op2 {| D = {  Operations.Multiply.Definition with Name = sprintf "%s%d" Operations.Multiply.Definition.Name id }
                       Arg0 = Arg {| Id = sprintf "W%d" id; TrackGradient = true |}
                       Arg1 = hg |}
              Arg1 = Arg {| Id = sprintf "b%d" id; TrackGradient = true |} |}

    g

  let private _makePredictGraph lossLayer g =
    let d =
      match lossLayer with
      | BCEWithLogitsLossLayer l ->
        { Activations.Sigmoid.Definition with Name = sprintf "%s[Predict,%d]" Activations.Sigmoid.Definition.Name l.Classes }
      | CCEWithLogitsLossLayer l ->
        { Activations.HardMax.Definition with Name = sprintf "%s[Predict,%d]" Activations.HardMax.Definition.Name l.Classes }
      | MSELossLayer ->
        { Activations.Linear.Definition with Name = sprintf "%s[Predict,%d]" Activations.Linear.Definition.Name 1 }

    Op1 {| D = d
           Arg = g |}

  let private _makeLossGraph lossLayer g =
    let d =
      match lossLayer with
      | BCEWithLogitsLossLayer l ->
        { LossFunctions.BCEWithLogitsLoss.Definition with Name = sprintf "%s[Loss,%d]" LossFunctions.BCEWithLogitsLoss.Definition.Name l.Classes }
      | CCEWithLogitsLossLayer l ->
        { LossFunctions.CCEWithLogitsLoss.Definition with Name = sprintf "%s[Loss,%d]" LossFunctions.CCEWithLogitsLoss.Definition.Name l.Classes }
      | MSELossLayer ->
        { LossFunctions.MSELoss.Definition with Name = sprintf "%s[Loss,%d]" LossFunctions.MSELoss.Definition.Name 1 }

    Op2 {| D = d
           Arg0 = Arg {| Id = "Y"; TrackGradient = false |}
           Arg1 = g |}

  let makeLayers seed heScale (inputLayer: InputLayer) (hiddenLayers: HiddenLayer list) (lossLayer: LossLayer): Net =
    let cg = _createCommonComputationGraph hiddenLayers

    let pg = _makePredictGraph lossLayer cg
    let lg = _makeLossGraph lossLayer cg

    let layerSizes = [ inputLayer.N ] @ (hiddenLayers |> List.map (fun l -> l.N)) @ [ lossLayer.Classes ]
    let initTensors = Tensor.createRandomizedR2 heScale
    let ps = initializeParameters seed initTensors layerSizes

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
