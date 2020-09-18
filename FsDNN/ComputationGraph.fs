namespace KS.FsDNN

[<AutoOpen>]
module ComputationGraphDomain =

  /// NOTE: Currently represent only feedforward neural networks.
  type ComputationGraph2<'TData, 'TOp1, 'TOp2> =
    | Arg of {| Id: string; TrackGradient: bool |}
    | Op1 of {| Id: string; Op: 'TOp1; Arg: ComputationGraph2<'TData, 'TOp1, 'TOp2>; |}
    | Op2 of {| Id: string; Op: 'TOp2; Arg0: ComputationGraph2<'TData, 'TOp1, 'TOp2>; Arg1: ComputationGraph2<'TData, 'TOp1, 'TOp2> |}

module ComputationGraph =
  let rec cata fArg fOp1 fOp2 (g: ComputationGraph2<'TData, 'TOp1, 'TOp2>): 'State =
    let recurse g = cata fArg fOp1 fOp2 g

    match g with
    | Arg a -> fArg a.Id
    | Op1 o -> fOp1 o.Id o.Op (recurse o.Arg)
    | Op2 o -> fOp2 o.Id o.Op (recurse o.Arg0) (recurse o.Arg1)

  let fold fArg fOp1 fOp2 (acc: 'State) (g: ComputationGraph2<'TData, 'TOp1, 'TOp2>): 'State =
    let rec loop t cont =
      match t with
      | Arg a -> cont (fArg a.Id acc)
      | Op1 o -> loop o.Arg (fun acc ->
                              cont (fOp1 o.Id o.Op acc))
      | Op2 o -> loop o.Arg0 (fun acc0 ->
                              loop o.Arg1 (fun acc1 ->
                              cont (fOp2 o.Id o.Op acc0 acc1)))

    loop g id

  let forward fArg fOp1 fOp2 (g: ComputationGraph2<'TData, 'TOperation1, 'TOperation2>) =
    let fArg' id =
      let ret = fArg id
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

  let rec backPropagate fOp1 fOp2 (grad0: 'TData) (g: ComputationGraph2<'TData, 'TOp1, 'TOp2>): Map<string, 'TData> =
    let recurse = backPropagate fOp1 fOp2

    match g with
    | Arg a ->
      if a.TrackGradient then Map.empty |> Map.add a.Id grad0 else Map.empty
    | Op1 o ->
      let outG = fOp1 grad0 o.Id o.Op
      let gradients = recurse outG o.Arg
      gradients
    | Op2 o ->
      let (outG0, outG1) = fOp2 grad0 o.Id o.Op
      let rGradients = recurse outG0 o.Arg0
      let lGradients = recurse outG1 o.Arg1
      rGradients |> Map.fold (fun acc key value -> Map.add key value acc) lGradients
