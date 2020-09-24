namespace KS.FsDNN

module LossFunctions =

  module BCEWithLogitsLoss =
    let private _forwardPropagate (Y: Tensor<double>) (Ŷ: Tensor<double>) =
      let c =
        Y.PointwiseMultiply(Ŷ.PointwiseLog()) +
        Y.Negate().Add(1.).PointwiseMultiply(Ŷ.Negate().Add(1.).PointwiseLog())

      let m = double Y.ColumnCount

      let it = TensorR0 ((-1. / m) * c.Sum())
      [| it; Y; Ŷ |]

    let private _backPropagate (cache: Cache<Tensor<double>>) id (inG: Tensor<double>) =
      let g0 = inG
      let Y = cache.[id].[1]
      let Ŷ = cache.[id].[2]
      let m = double Y.ColumnCount
      let g1 = (Y.PointwiseDivide(Ŷ.Add(Constants.DivideBy0Guard)).Negate() + Y.Negate().Add(1.0).PointwiseDivide(Ŷ.Negate().Add(1.0 + Constants.DivideBy0Guard))).PointwiseDivide(TensorR0 m)
      (g0, inG.PointwiseMultiply(g1))

    let Definition: Operation2Definition<_> =
      { Name = "BCEWithLogitsLoss"
        Functions = { F = _forwardPropagate; B = _backPropagate } }

  module CCEWithLogitsLoss =
    let private _forwardPropagate (Y: Tensor<double>) (Ŷ: Tensor<double>)=
      let softMax = Activations.SoftMax.Definition.Functions.F Ŷ |> Array.head
      let it = (Y.Negate().PointwiseMultiply(softMax.PointwiseLog())).Sum() |> TensorR0
      [| it; Y; Ŷ |]

    let private _backPropagate (cache: Cache<Tensor<double>>) id (inG: Tensor<double>) =
      let g0 = inG
      let Y = cache.[id].[1]
      let Ŷ = cache.[id].[2]
      // TODO: Get this from the cache
      let softMax = Activations.SoftMax.Definition.Functions.F Ŷ |> Array.head
      let g1 = softMax - Y
      (g0, inG.PointwiseMultiply(g1))

    let Definition: Operation2Definition<_> =
      { Name = "CCEWithLogitsLoss"
        Functions = { F = _forwardPropagate; B = _backPropagate } }

  module MSELoss =
    let private _forwardPropagate (Y: Tensor<double>) (Ŷ: Tensor<double>) =
      let c = (Ŷ - Y).PointwisePower(2.)
      let m = double Y.ColumnCount

      let it = TensorR0 ((1. / (2. * m)) * c.Sum())

      [| it; Y; Ŷ |]

    let private _backPropagate (cache: Cache<Tensor<double>>) id (inG: Tensor<double>) =
      let g0 = inG
      let Y = cache.[id].[1]
      let Ŷ = cache.[id].[2]
      let m = double Y.ColumnCount
      let g1 = (Ŷ - Y).PointwiseDivide(TensorR0 m)
      (g0, inG.PointwiseMultiply(g1))

    let Definition: Operation2Definition<_> =
      { Name = "MSELoss"
        Functions = { F = _forwardPropagate; B = _backPropagate } }
