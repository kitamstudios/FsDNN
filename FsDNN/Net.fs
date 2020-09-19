namespace KS.FsDNN

open MathNet.Numerics.Distributions
open MathNet.Numerics.LinearAlgebra
open System

[<AutoOpen>]
module NetDomain =

  type InputLayer =
    { N: int }

  type Layer =
    | FullyConnected of {| N: int |}

  type LossLayer =
    | SoftMax of {| Classes: int |}

    member this.N =
      match this with
      | SoftMax s -> s.Classes

  type Operations1 =
    | OpSigmoid

  type Operations2 =
    | OpAdd
    | OpMultiply
    | OpL1Loss

  type Net =
    { LossGraph: ComputationGraph<Matrix<double>, Operations1, Operations2>
      PredictGraph: ComputationGraph<Matrix<double>, Operations1, Operations2>
      Parameters: Map<string, Matrix<double>> }


// TODO
// - Dont calculate intermediate when just forwarding

module Net =

  let toString n: string =
    n.LossGraph |> ComputationGraph.toString

  let private _initializeParameters (seed: int) (heScale: double): Map<string, Matrix<double>> =

    let ws =
      seq { yield ("W", 1, 3) }
      |> Seq.mapi (fun i (id, n, nPrev) -> id, (DenseMatrix.random<double> n nPrev (Normal.WithMeanVariance(0.0, 1.0, Random(seed + i)))) * Math.Sqrt(2.0 / double nPrev) * heScale)

    let bs =
      seq { yield ("b", 1) }
      |> Seq.map (fun (id, n) -> id, (DenseMatrix.zero<double> n 1))

    seq { yield ws; yield bs }
    |> Seq.concat
    |> Map.ofSeq

  let makeLayers seed heScale (inputLayer: InputLayer) (hiddenLayers: Layer list) (lossLayer: LossLayer): Net =
    let pg = Op1 {| Id = "OpSigmoid"
                    Op = OpSigmoid
                    Arg =
                      Op2 {| Id = "OpAdd"
                             Op = OpAdd
                             Arg0 = Arg {| Id = "b"; TrackGradient = true |}
                             Arg1 =
                               Op2 {| Id = "OpMultiply"
                                      Op = OpMultiply
                                      Arg0 = Arg {| Id = "W"; TrackGradient = true |}
                                      Arg1 = Arg {| Id = "X"; TrackGradient = false |} |} |} |}

    let lg = Op2 {| Id = "L1Loss"
                    Op = OpL1Loss
                    Arg0 = Arg {| Id = "Y"; TrackGradient = false |}
                    Arg1 = pg |}

    let ps = _initializeParameters seed heScale

    { LossGraph = lg
      PredictGraph = pg
      Parameters = ps }

  let private _forwardArg parameters id: Matrix<double> =
    parameters |> Map.find id

  let private _forwardOp1 o (arg: Matrix<double>): Matrix<double> =
    match o with
    | OpSigmoid -> arg.Negate().PointwiseExp().Add(1.0).PointwisePower(-1.0)

  let private _forwardOp2 o (arg0: Matrix<double>) (arg1: Matrix<double>): Matrix<double> =
    match o with
    | OpAdd -> arg0 + arg1
    | OpMultiply -> arg0 * arg1
    | OpL1Loss -> arg0

  let predict (X: Matrix<double>) n =
    let parameters = n.Parameters |> Map.add "X" X
    ComputationGraph.predict (_forwardArg parameters) _forwardOp1 _forwardOp2 n.PredictGraph
