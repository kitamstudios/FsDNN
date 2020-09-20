namespace KS.FsDNN

open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.Distributions
open System
open System.Linq
open System.Runtime.CompilerServices

[<AutoOpen>]
module TensorDomain =
  type Tensor<'TData
    when 'TData: (new: unit -> 'TData)
     and 'TData: struct and 'TData :> IEquatable<'TData>
     and 'TData :> IFormattable
     and 'TData :> ValueType> =
    | R2 of Matrix<'TData>
    | R1 of Vector<'TData>
    | R0 of 'TData

    member this.ColumnCount =
      match this with
      | R2 m -> m.ColumnCount
      | R1 v -> v.Count
      | R0 _ -> 1

    static member inline Add(t0: Tensor<'TData>, t1: Tensor<'TData>): Tensor<'TData> =
      match (t0, t1) with
      | R2 m0, R2 m1 -> m0 + m1 |> R2
      | R1 v0, R2 m1 -> DenseMatrix.ofColumnSeq (Enumerable.Repeat(v0, m1.ColumnCount)) + m1 |> R2
      | R2 m0, R0 s1 -> m0.Add(s1) |> R2
      | _ -> Prelude.undefined

    static member inline Subtract(t0: Tensor<'TData>, t1: Tensor<'TData>): Tensor<'TData> =
      match (t0, t1) with
      | R2 m0, R2 m1 -> m0 - m1 |> R2
      | R0 s0, R2 m1 -> m1.Negate().Add(s0) |> R2
      | _ -> Prelude.undefined

    static member inline Multiply(t0: Tensor<'TData>, t1: Tensor<'TData>): Tensor<'TData> =
      match (t0, t1) with
      | R2 m0, R2 m1 -> m0 * m1 |> R2
      | R2 m0, R0 s1 -> m0.Multiply(s1) |> R2
      | R0 s0, R2 m1 -> m1.Multiply(s0) |> R2
      | _ -> Prelude.undefined

    static member inline Divide(t0: Tensor<'TData>, t1: Tensor<'TData>): Tensor<'TData> =
      match (t0, t1) with
      | R2 m0, R2 m1 -> m0.PointwiseDivide(m1) |> R2
      | R2 m0, R0 s1 -> m0.Divide(s1) |> R2
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
    | R2 m -> m.Negate() |> R2
    | _ -> Prelude.undefined

  [<Extension>]
  static member inline PointwiseExp(t: Tensor<'TData>): Tensor<'TData> =
    match t with
    | R2 m -> m.PointwiseExp() |> R2
    | _ -> Prelude.undefined

  [<Extension>]
  static member inline PointwisePower(t: Tensor<'TData>, exponent: 'TData): Tensor<'TData> =
    match t with
    | R2 m -> m.PointwisePower(exponent) |> R2
    | _ -> Prelude.undefined

  [<Extension>]
  static member inline Add(t0: Tensor<'TData>, x: 'TData): Tensor<'TData> =
    t0 + (x |> R0)

  [<Extension>]
  static member inline PointwiseMultiply(t0: Tensor<'TData>, t1: Tensor<'TData>): Tensor<'TData> =
    match t0, t1 with
    | R2 m0, R2 m1 -> m0.PointwiseMultiply(m1) |> R2
    | _ -> Prelude.undefined

module Tensor =
  let createRandomizedR2 seed rows cols scale =
    let t0 = DenseMatrix.random<double> rows cols (Normal.WithMeanVariance(0.0, 1.0, Random(seed))) |> R2
    let t1 = Math.Sqrt(2. / double cols) * scale |> R0
    t0 * t1

  let createZerosR1 n =
    DenseVector.zero<double> n |> R1

  let ofListOfList (rs: double list list) = rs |> array2D |> CreateMatrix.DenseOfArray |> R2
