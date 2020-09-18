namespace KS.FsDNN

(*
- Assume
  - Binary tree with optional right child
  - No splits

- J = ((x + y) * z) ^ 2
  - forward
  - backPropagate
- get mutable out of core
  - Intermediate data
  - Gradients
  - clean up tests + code
  - rename tests and code
  - attempt to merge with fold
  - unused variables
  - Update parameters with track gradients
  - remove .Data and pass data map as well
- generalize for matrix
- generalize functions
- track gradient flag


- recompute intermediate values
  - remove mutable inx
  - remove separate declarations of type

- recompute gradients
- update gradients
- allow opting out of gradient updates


 *)

[<AutoOpen>]
module ComputationGraphDomain =

  /// NOTE: Currently represent only feedforward neural networks.
  type ComputationGraph2<'TData, 'TOp1, 'TOp2> =
    | Arg of {| Id: string; Data: 'TData; |}
    | Op1 of {| Id: string; Op: 'TOp1; Arg: ComputationGraph2<'TData, 'TOp1, 'TOp2>; |}
    | Op2 of {| Id: string; Op: 'TOp2; Arg0: ComputationGraph2<'TData, 'TOp1, 'TOp2>; Arg1: ComputationGraph2<'TData, 'TOp1, 'TOp2> |}

module ComputationGraph =
  let rec cata fArg fOp1 fOp2 (g: ComputationGraph2<'TData, 'TOp1, 'TOp2>): 'State =
    let recurse g = cata fArg fOp1 fOp2 g

    match g with
    | Arg a -> fArg a.Id a.Data
    | Op1 o -> fOp1 o.Id o.Op (recurse o.Arg)
    | Op2 o -> fOp2 o.Id o.Op (recurse o.Arg0) (recurse o.Arg1)

  let fold fArg fOp1 fOp2 (acc: 'State) (g: ComputationGraph2<'TData, 'TOp1, 'TOp2>): 'State =
    let rec loop t cont =
      match t with
      | Arg a -> cont (fArg a.Id a.Data acc)
      | Op1 o -> loop o.Arg (fun acc ->
                              cont (fOp1 o.Id o.Op acc))
      | Op2 o -> loop o.Arg0 (fun acc0 ->
                              loop o.Arg1 (fun acc1 ->
                              cont (fOp2 o.Id o.Op acc0 acc1)))

    loop g id

  let forward fArg fOp1 fOp2 (g: ComputationGraph2<'TData, 'TOperation1, 'TOperation2>) =
    let fArg' _ value =
      let ret = fArg value
      ret, Map.empty

    let fOp1' id op (arg1, m) =
      let ret = fOp1 op arg1
      let m = m |> Map.add id [| arg1 |]
      ret, m

    let fOp2' id op (arg0, m0) (arg1, m1) =
      let ret = fOp2 op arg0 arg1
      let m = Map.fold (fun acc key value -> Map.add key value acc) m0 m1
      let m = m |> Map.add id [| arg0; arg1 |]
      ret, m

    cata fArg' fOp1' fOp2' g

  let rec backPropagate fArg fOp1 fOp2 (grad0: 'TData, iValues: Map<string, 'TData[]>, gradients: Map<string, 'TData>) (g: ComputationGraph2<'TData, 'TOp1, 'TOp2>): 'TData * Map<string, 'TData[]> * Map<string, 'TData> =
    let recurse = backPropagate fArg fOp1 fOp2

    match g with
    | Arg a ->
      fArg (grad0, iValues, gradients) a.Id a.Data
    | Op1 o ->
      let outG = fOp1 (grad0, iValues) o.Id o.Op
      let (grad0'', iValues, gradients'') = recurse (outG, iValues, gradients) o.Arg
      (grad0, iValues, gradients'')
    | Op2 o ->
      let (outG0, outG1) = fOp2 (grad0, iValues) o.Id o.Op
      let (grad0'', iValues, gradients') = recurse (outG0, iValues, gradients) o.Arg0
      let (grad0''', iValues, gradients'') = recurse (outG1, iValues, gradients') o.Arg1
      (grad0, iValues, gradients'')
