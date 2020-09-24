namespace KS.FsDNN.Tests

open KS.FsDNN

[<AutoOpen>]
module TestHelpers =

  open FluentAssertions
  open FluentAssertions.Equivalency
  open System

  [<Literal>]
  let precision = 5e-7;

  let doubleComparisonOptions<'TExpectation> (o: EquivalencyAssertionOptions<'TExpectation>): EquivalencyAssertionOptions<'TExpectation> =
    let action = fun (ctx: IAssertionContext<double>) -> ctx.Subject.Should().BeApproximately(ctx.Expectation, precision, String.Empty, Array.Empty<obj>()) |> ignore
    o.Using<double>(Action<IAssertionContext<double>>(action)).WhenTypeIs<double>()

  let shouldBeEquivalentTo (a: double list list) (m: Tensor<double>) =
    let a' =
      match m with
      | TensorR2 m -> m.ToArray()
      | TensorR1 v -> array2D [| v.ToArray() |]
      | TensorR0 s -> array2D [| [| s |] |]

    a'.Should().BeEquivalentTo(array2D a, doubleComparisonOptions, String.Empty, Array.empty) |> ignore

  let shouldBeEquivalentToT (m2: Tensor<double>) (m1: Tensor<double>) =
    let a1, a2 =
      match m1, m2 with
      | TensorR2 m1, TensorR2 m2 -> m1.ToArray(), m2.ToArray()
      | TensorR1 v1, TensorR1 v2 -> array2D [| v1.ToArray() |], array2D [| v2.ToArray() |]
      | TensorR0 s1, TensorR0 s2 -> array2D [| [| s1 |] |], array2D [| [| s2 |] |]
      | _, _ -> Prelude.undefined

    a1.Should().BeEquivalentTo(a2, doubleComparisonOptions, String.Empty, Array.empty) |> ignore

  let shouldBeEquivalentToT2 (m2s: Tensor<double>[]) (m1s: Tensor<double>[]) =
    let _f i m1 m2 =
      let a1, a2 =
        match m1, m2 with
        | TensorR2 m1, TensorR2 m2 -> m1.ToArray(), m2.ToArray()
        | TensorR1 v1, TensorR1 v2 -> array2D [| v1.ToArray() |], array2D [| v2.ToArray() |]
        | TensorR0 s1, TensorR0 s2 -> array2D [| [| s1 |] |], array2D [| [| s2 |] |]
        | _, _ -> Prelude.undefined

      a1.Should().BeEquivalentTo(a2, doubleComparisonOptions, "Tensors at index {0} are not equal", [| i |]) |> ignore

    Array.iteri2 _f m2s m1s


  let shouldBeApproximately (a1: double) (a2) =
    a1.Should().BeApproximately(a2, precision, String.Empty, Array.empty) |> ignore
