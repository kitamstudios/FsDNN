module KS.FsDNN.Tests.Loader

open KS.FsDNN
open Xunit
open FsUnit.Xunit

[<Fact>]
let ``Check getBatches for 1``() =
  let X = Tensor.createWithInit 100 100 (fun i j -> float i * 10.0 + float j)
  let Y = Tensor.createWithInit  4   100 (fun i j -> float i * 20.0 + float j)
  let batches = Loader.getBatches BatchSize1 (X, Y)

  batches |> Seq.length |> should equal 100

  let X0, Y0 = batches |> Seq.skip 27 |> Seq.take 1 |> Seq.exactlyOne
  X0 |> shouldBeEquivalentToT (Tensor.createWithInit 100 1 (fun i _ -> float i * 10.0 + 27.))
  Y0 |> shouldBeEquivalentTo [[27.]; [47.]; [67.]; [87.]]

  let X0, Y0 = batches |> Seq.skip 99 |> Seq.take 1 |> Seq.exactlyOne
  X0 |> shouldBeEquivalentToT (Tensor.createWithInit 100 1 (fun i _ -> float i * 10.0 + 99.))
  Y0 |> shouldBeEquivalentTo [[99.]; [119.]; [139.]; [159.]]

[<Fact>]
let ``Check getBatches - complete``() =
  let X = Tensor.createWithInit 10 128 (fun i j -> float i * 10.0 + float j)
  let Y = Tensor.createWithInit 3  128 (fun i j -> float i * 20.0 + float j)
  let batches = Loader.getBatches BatchSize64 (X, Y)

  batches |> Seq.length |> should equal 2

  let X0, Y0 = batches |> Seq.skip 0 |> Seq.take 1 |> Seq.exactlyOne
  X0 |> shouldBeEquivalentToT (Tensor.createWithInit 10 64 (fun i j -> float i * 10.0 + float j))
  Y0 |> shouldBeEquivalentToT (Tensor.createWithInit 3  64 (fun i j -> float i * 20.0 + float j))

  let X0, Y0 = batches |> Seq.skip 1 |> Seq.take 1 |> Seq.exactlyOne
  X0 |> shouldBeEquivalentToT (Tensor.createWithInit 10 64 (fun i j -> float i * 10.0 + float j + 64.))
  Y0 |> shouldBeEquivalentToT (Tensor.createWithInit 3  64 (fun i j -> float i * 20.0 + float j + 64.))

[<Fact>]
let ``Check getBatches - partial``() =
  let X = Tensor.createWithInit 11 100 (fun i j -> float i * 10.0 + float j)
  let Y = Tensor.createWithInit 5  100 (fun i j -> float i * 20.0 + float j)
  let batches = Loader.getBatches BatchSize128 (X, Y)

  batches |> Seq.length |> should equal 1

  let X0, Y0 = batches |> Seq.skip 0 |> Seq.take 1 |> Seq.exactlyOne
  X0 |> shouldBeEquivalentToT (Tensor.createWithInit 11 100 (fun i j -> float i * 10.0 + float j))
  Y0 |> shouldBeEquivalentToT (Tensor.createWithInit 5  100 (fun i j -> float i * 20.0 + float j))

[<Fact>]
let ``Check getBatches - complete and partial``() =
  let X = Tensor.createWithInit 11 300 (fun i j -> float i * 10.0 + float j)
  let Y = Tensor.createWithInit 5  300 (fun i j -> float i * 20.0 + float j)
  let batches = Loader.getBatches BatchSize256 (X, Y)

  batches |> Seq.length |> should equal 2

  let X0, Y0 = batches |> Seq.skip 0 |> Seq.take 1 |> Seq.exactlyOne
  X0 |> shouldBeEquivalentToT (Tensor.createWithInit 11 256 (fun i j -> float i * 10.0 + float j))
  Y0 |> shouldBeEquivalentToT (Tensor.createWithInit 5  256 (fun i j -> float i * 20.0 + float j))

  let X1, Y1 = batches |> Seq.skip 1 |> Seq.take 1 |> Seq.exactlyOne
  X1 |> shouldBeEquivalentToT (Tensor.createWithInit 11 44 (fun i j -> float i * 10.0 + float j + 256.))
  Y1 |> shouldBeEquivalentToT (Tensor.createWithInit 5  44 (fun i j -> float i * 20.0 + float j + 256.))

[<Fact>]
let ``Check getBatches - all``() =
  let X = Tensor.createWithInit 17 64 (fun i j -> float i * 10.0 + float j)
  let Y = Tensor.createWithInit 3  64 (fun i j -> float i * 20.0 + float j)
  let batches = Loader.getBatches BatchSizeAll (X, Y)

  batches |> Seq.length |> should equal 1

  let X0, Y0 = batches |> Seq.skip 0 |> Seq.take 1 |> Seq.exactlyOne
  X0 |> shouldBeEquivalentToT (Tensor.createWithInit 17 64 (fun i j -> float i * 10.0 + float j))
  Y0 |> shouldBeEquivalentToT (Tensor.createWithInit 3  64 (fun i j -> float i * 20.0 + float j))
