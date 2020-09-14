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

    member this.N =
      match this with
      | SoftMax s -> s.nc

  type TransformerInfo<'TData> =
    { Data: 'TData
      Forward: Matrix<double> -> Matrix<double> }

  type GenericTransformerData =
    { W: Matrix<double>
      b: Matrix<double> }

  type Transformer =
    | InputTransformer of TransformerInfo<unit>
    | GenericTransformer of TransformerInfo<GenericTransformerData>
    | LossTransformer of TransformerInfo<unit>

    member this.Forward =
      match this with
      | InputTransformer t -> t.Forward
      | GenericTransformer t -> t.Forward
      | LossTransformer t -> t.Forward

  type NetX =
    { Transformers: Transformer list }
