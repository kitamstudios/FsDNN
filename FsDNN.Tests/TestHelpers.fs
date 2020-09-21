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
      | R2 m -> m.ToArray()
      | R1 v -> array2D [| v.ToArray() |]
      | R0 s -> array2D [| [| s |] |]

    a'.Should().BeEquivalentTo(array2D a, doubleComparisonOptions, String.Empty, Array.empty) |> ignore

  let shouldBeApproximately (a1: double) (a2) =
    a1.Should().BeApproximately(a2, precision, String.Empty, Array.empty) |> ignore
