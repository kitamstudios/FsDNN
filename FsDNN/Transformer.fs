namespace KS.FsDNN

module Transformer =

  open System
  open MathNet.Numerics.Distributions
  open MathNet.Numerics.LinearAlgebra

  module Input =
    let create () =
      InputTransformer { Data = (); Forward = id }

  module FullyConnected =

    let create seed nPrev n =
      let W = DenseMatrix.random<double> n nPrev (Normal(0., 1., Random(seed)))
      let b = DenseMatrix.create n 1 0.

      GenericTransformer { Data = { W = W; b = b }; Forward = fun a -> W * a + b }

  module Loss =
    module SoftMax =

      let forward (x: Matrix<double>) =
        let exp = x.PointwiseExp();
        exp.Divide(exp.ColumnSums().Sum() + Constants.DivideBy0Guard)

      let create () =
        LossTransformer { Data = (); Forward = forward }

    let create = function
      SoftMax _ -> SoftMax.create ()
