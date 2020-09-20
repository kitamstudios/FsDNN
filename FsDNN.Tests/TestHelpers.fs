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

  let shouldBeEquivalent (a: double list list) (m: Tensor<double>) =
    match m with
    | R2 m -> m.ToArray().Should().BeEquivalentTo(array2D a, doubleComparisonOptions, String.Empty, Array.empty) |> ignore
    | R1 v -> (array2D [| v.ToArray() |]).Should().BeEquivalentTo(array2D a, doubleComparisonOptions, String.Empty, Array.empty) |> ignore
    | _ -> Prelude.undefined


  let shouldBeApproximately (a1: double) (a2) =
    a1.Should().BeApproximately(a2, precision, String.Empty, Array.empty) |> ignore
