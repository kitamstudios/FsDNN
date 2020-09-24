namespace KS.FsDNN

module Activations =

  module Sigmoid =
    let private _forwardPropagate (arg: Tensor<double>) =
      let it = arg.Negate().PointwiseExp().Add(1.0).PointwisePower(-1.0)
      [| it; arg |]

    let private _backPropagate (cache: Cache<Tensor<double>>) id (inG: Tensor<double>) =
      let s = cache.[id].[0]
      inG.PointwiseMultiply(s.PointwiseMultiply(s.Negate().Add(1.0)))

    let Definition: Operation1Definition<_> =
      { Name = "Sigmoid"
        Functions = { F = _forwardPropagate; B = _backPropagate } }

  module Linear =
    let Definition: Operation1Definition<_> =
      { Name = "Linear"
        Functions = { F = (fun x -> [| x |] ); B = fun _ _ -> id } }

  module SoftMax =
    let private _forwardPropagate (arg: Tensor<double>) =
      let exp = arg.PointwiseExp()
      let it = exp / exp.ColumnSums()

      [| it; arg |]

    let Definition: Operation1Definition<_> =
      { Name = "SoftMax"
        Functions = { F = _forwardPropagate; B = fun _ _ -> Prelude.undefined } }

  module HardMax =
    let private _forwardPropagate (arg: Tensor<double>) =
      let it = arg.ColumnHardMax()
      [| it; arg |]

    let Definition: Operation1Definition<_> =
      { Name = "HardMax"
        Functions = { F = _forwardPropagate; B = fun _ _ -> Prelude.undefined } }
