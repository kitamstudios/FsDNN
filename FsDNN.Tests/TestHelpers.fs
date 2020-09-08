namespace KS.FsDNN.Tests

[<AutoOpen>]
module TestHelpers =

  open FluentAssertions
  open FluentAssertions.Equivalency
  open System
  open MathNet.Numerics.LinearAlgebra

  [<Literal>]
  let precision = 5e-7;

  let doubleComparisonOptions<'TExpectation> (o: EquivalencyAssertionOptions<'TExpectation>): EquivalencyAssertionOptions<'TExpectation> =
    let action = fun (ctx: IAssertionContext<double>) -> ctx.Subject.Should().BeApproximately(ctx.Expectation, precision, String.Empty, Array.Empty<obj>()) |> ignore
    o.Using<double>(Action<IAssertionContext<double>>(action)).WhenTypeIs<double>()

  let shouldBeEquivalent (a: double list list) (m: Matrix<double>) =
    m.ToArray().Should().BeEquivalentTo(array2D a, doubleComparisonOptions, String.Empty, Array.empty) |> ignore

  let shouldBeEquivalentM2 (a: double [,]) (m: Matrix<double>) =
    m.ToArray().Should().BeEquivalentTo(a, doubleComparisonOptions, String.Empty, Array.empty) |> ignore

  let shouldBeApproximately (a1: double) (a2) =
    a1.Should().BeApproximately(a2, precision, String.Empty, Array.empty) |> ignore

  let toM (rs: double list list) = rs |> array2D |> CreateMatrix.DenseOfArray
