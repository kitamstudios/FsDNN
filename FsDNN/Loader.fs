namespace KS.FsDNN

open System
open System.Diagnostics

[<AutoOpen>]
module LoaderDomain =
  type BatchSize =
    | BatchSize1
    | BatchSize64
    | BatchSize128
    | BatchSize256
    | BatchSize512
    | BatchSize1024
    | BatchSizeAll
    with
      member this.toInt max =
        match this with
        | BatchSize1 -> 1
        | BatchSize64 -> 64
        | BatchSize128 -> 128
        | BatchSize256 -> 256
        | BatchSize512 -> 512
        | BatchSize1024 -> 1024
        | BatchSizeAll -> max

module Loader =

  let getBatches (batchSize: BatchSize) (X: Tensor<double>, Y: Tensor<double>): (Tensor<double> * Tensor<double>) seq =
    if X.ColumnCount <> Y.ColumnCount then
      failwithf "Column count of X (%d) is not the same as that of Y (%d)." X.ColumnCount Y.ColumnCount

    if batchSize = BatchSizeAll then
      seq {
        yield X, Y
      }
    else
      let bs = batchSize.toInt X.ColumnCount
      seq {
        let nComplete = X.ColumnCount / bs
        for bn = 0 to nComplete - 1 do
          let c0 = bn * bs
          let c1 = c0 + bs - 1
          yield X.[*, c0 .. c1], Y.[*, c0 .. c1]

        if X.ColumnCount % bs <> 0 then
          let c0 = nComplete * bs
          let c1 = X.ColumnCount - 1
          yield X.[*, c0 .. c1], Y.[*, c0 .. c1]
      }

