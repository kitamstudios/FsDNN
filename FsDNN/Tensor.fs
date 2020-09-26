namespace KS.FsDNN

open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.Distributions
open System
open System.Linq
open System.Runtime.CompilerServices

[<AutoOpen>]
module TensorDomain =
  ///
  /// Emulates np.array by providing an unified interface over Math.Net Matrix and Vector.
  ///
  /// NOTE:
  /// - Broadcasting support is limited.
  ///   - Ensure first parameter has target dimensions.
  ///   - Scalar arguments must be first and will be treated as point-wise operations. i.e. no broadcasting.
  ///
  type Tensor<'TData
    when 'TData : (new: unit -> 'TData)
     and 'TData : struct and 'TData :> IEquatable<'TData>
     and 'TData :> IFormattable
     and 'TData :> ValueType
     > =
    | TensorR2 of Matrix<'TData>
    | TensorR1 of Vector<'TData>
    | TensorR0 of 'TData

    member inline this.ColumnCount =
      match this with
      | TensorR2 m -> m.ColumnCount
      | TensorR1 _ -> 1
      | TensorR0 _ -> 1

    member inline this.RowCount =
      match this with
      | TensorR2 m -> m.RowCount
      | TensorR1 v -> v.Count
      | TensorR0 _ -> 1

    member inline this.Shape =
      (this.RowCount, this.ColumnCount)

    static member inline Add(t0: Tensor<'TData>, t1: Tensor<'TData>): Tensor<'TData> =
      match (t0, t1) with
      | TensorR2 m0, TensorR2 m1 -> (m0 + m1) |> TensorR2
      | TensorR2 m0, TensorR1 v1 -> (m0 + DenseMatrix.ofColumnSeq (Enumerable.Repeat(v1, m0.ColumnCount))) |> TensorR2
      | TensorR2 m0, TensorR0 s1 -> m0.Add(s1) |> TensorR2
      | _ -> Prelude.undefined

    static member inline Subtract(t0: Tensor<'TData>, t1: Tensor<'TData>): Tensor<'TData> =
      match (t0, t1) with
      | TensorR2 m0, TensorR2 m1 -> m0.Subtract(m1) |> TensorR2
      | TensorR1 v0, TensorR2 m1 -> (v0 - (m1.EnumerateColumns() |> Seq.reduce (+))) |> TensorR1
      | _ -> Prelude.undefined

    static member inline Multiply(t0: Tensor<'TData>, t1: Tensor<'TData>): Tensor<'TData> =
      match (t0, t1) with
      | TensorR2 m0, TensorR2 m1 -> m0.Multiply(m1) |> TensorR2
      | TensorR2 m0, TensorR0 s1 -> m0.Multiply(s1) |> TensorR2
      | TensorR0 s0, TensorR2 m1 -> m1.Multiply(s0) |> TensorR2
      | _ -> Prelude.undefined

    static member inline Divide(t0: Tensor<'TData>, t1: Tensor<'TData>): Tensor<'TData> =
      match (t0, t1) with
      | TensorR2 m0, TensorR2 m1 -> m0.PointwiseDivide(m1) |> TensorR2
      | TensorR2 m0, TensorR1 v1 -> (m0.PointwiseDivide(DenseMatrix.ofRowSeq (Enumerable.Repeat(v1, m0.RowCount)))) |> TensorR2
      | TensorR2 m0, TensorR0 s1 -> m0.Divide(s1) |> TensorR2
      | _ -> Prelude.undefined

    static member inline ( + ) (t0, t1): Tensor<'TData> = Tensor<'TData>.Add(t0, t1)

    static member inline ( - ) (t0, t1): Tensor<'TData> = Tensor<'TData>.Subtract(t0, t1)

    static member inline ( * ) (t0, t1): Tensor<'TData> = Tensor<'TData>.Multiply(t0, t1)

    static member inline ( / ) (t0, t1): Tensor<'TData> = Tensor<'TData>.Divide(t0, t1)

[<Extension>]
type Tensor =
  [<Extension>]
  static member inline Negate(t: Tensor<'TData>): Tensor<'TData> =
    match t with
    | TensorR2 m -> m.Negate() |> TensorR2
    | _ -> Prelude.undefined

  [<Extension>]
  static member inline PointwiseExp(t: Tensor<'TData>): Tensor<'TData> =
    match t with
    | TensorR2 m -> m.PointwiseExp() |> TensorR2
    | _ -> Prelude.undefined

  [<Extension>]
  static member inline PointwisePower(t: Tensor<'TData>, exponent: 'TData): Tensor<'TData> =
    match t with
    | TensorR2 m -> m.PointwisePower(exponent) |> TensorR2
    | _ -> Prelude.undefined

  [<Extension>]
  static member inline Add(t0: Tensor<'TData>, x: 'TData): Tensor<'TData> =
    t0 + (x |> TensorR0)

  [<Extension>]
  static member inline PointwiseMultiply(t0: Tensor<'TData>, t1: Tensor<'TData>): Tensor<'TData> =
    match t0, t1 with
    | TensorR2 m0, TensorR2 m1 -> m0.PointwiseMultiply(m1) |> TensorR2
    | TensorR0 s0, TensorR2 m1 -> m1.Multiply(s0) |> TensorR2
    | TensorR0 s0, TensorR0 s1 -> (s0 * s1) |> TensorR0
    | _ -> Prelude.undefined

  [<Extension>]
  static member inline PointwiseDivide(t0: Tensor<'TData>, t1: Tensor<'TData>): Tensor<'TData> =
    match t0, t1 with
    | TensorR2 m0, TensorR2 m1 -> m0.PointwiseDivide(m1) |> TensorR2
    | TensorR2 m0, TensorR0 s1 -> m0.Divide(s1) |> TensorR2
    | _ -> Prelude.undefined

  [<Extension>]
  static member inline TransposeAndMultiply(t0: Tensor<'TData>, t1: Tensor<'TData>): Tensor<'TData> =
    match t0, t1 with
    | TensorR2 m0, TensorR2 m1 -> m0.TransposeAndMultiply(m1) |> TensorR2
    | _ -> Prelude.undefined

  [<Extension>]
  static member inline TransposeThisAndMultiply(t0: Tensor<'TData>, t1: Tensor<'TData>): Tensor<'TData> =
    match t0, t1 with
    | TensorR2 m0, TensorR2 m1 -> m0.TransposeThisAndMultiply(m1) |> TensorR2
    | _ -> Prelude.undefined

  [<Extension>]
  static member inline PointwiseLog(t: Tensor<'TData>): Tensor<'TData> =
    match t with
    | TensorR2 m -> m.PointwiseLog() |> TensorR2
    | _ -> Prelude.undefined

  [<Extension>]
  static member inline PointwiseMaximum((t: Tensor<'TData>), (other: 'TData)): Tensor<'TData> =
    match t with
    | TensorR2 m -> m.PointwiseMaximum(other) |> TensorR2
    | _ -> Prelude.undefined

  [<Extension>]
  static member inline PointwiseSign(t: Tensor<'TData>): Tensor<'TData> =
    match t with
    | TensorR2 m -> m.PointwiseSign() |> TensorR2
    | _ -> Prelude.undefined

  [<Extension>]
  static member inline Sum(t: Tensor<'TData>): 'TData =
    match t with
    | TensorR2 m -> m.ColumnSums().Sum()
    | TensorR1 v -> v.Sum()
    | TensorR0 s -> s

  [<Extension>]
  static member inline ColumnSums(t: Tensor<'TData>): Tensor<'TData> =
    match t with
    | TensorR2 m -> m.ColumnSums() |> TensorR1
    | _ -> Prelude.undefined

  [<Extension>]
  static member inline ColumnHardMax(t: Tensor<'TData>): Tensor<'TData> =
    match t with
    | TensorR2 m ->
      let mapVector _ v: Vector<'TData> =
        let imax = Vector.maxIndex v
        v |> Vector.mapi (fun i _ -> if i = imax then Matrix.One else Matrix.Zero)
      m |> Matrix.mapCols mapVector |> TensorR2
    | _ -> Prelude.undefined

module Tensor =
  let createRandomizedR2 scale seed rows cols =
    let t0 = DenseMatrix.random<double> rows cols (Normal.WithMeanVariance(0.0, 1.0, Random(seed))) |> TensorR2
    let t1 = Math.Sqrt(2. / double cols) * scale |> TensorR0
    t0 * t1

  let createZerosR1 n =
    DenseVector.zero<double> n |> TensorR1

  let ofListOfList (rs: double list list) = rs |> array2D |> CreateMatrix.DenseOfArray |> TensorR2

  let ofList (rs: double list) = rs |> Array.ofSeq |> CreateVector.DenseOfArray |> TensorR1
