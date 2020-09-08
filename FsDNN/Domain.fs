namespace KS.FsDNN

open MathNet.Numerics.LinearAlgebra

[<AutoOpen>]
module Domain =

  type InputLayer =
    { nx: int }

  type Layer =
    | FullyConnected of {| n: int |}

  type LossLayer =
    | SoftMax of {| nc: int |}

  type Transformer =
    { n: int
      W: Matrix<double>
      b: Matrix<double>
      forward: Matrix<double> -> Matrix<double> }

  type Net =
    { Transformers: Transformer list }
