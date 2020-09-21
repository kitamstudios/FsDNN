namespace KS.FsDNN

[<AutoOpen>]
module ComputationGraphDomain =

  type Op1Forward<'TData> = 'TData -> 'TData

  type Op2Forward<'TData> = 'TData -> 'TData -> 'TData

  type Op1BackPropagate<'TData> = ('TData -> string -> 'TData)

  type Op2BackPropagate<'TData> = ('TData -> string -> 'TData * 'TData)


  type ComputationGraph<'TData, 'TOp1, 'TOp2> =
    | Arg of {| Id: string; TrackGradient: bool |}
    | Op1 of {| Id: string; Op: 'TOp1; Arg: ComputationGraph<'TData, 'TOp1, 'TOp2>; |}
    | Op2 of {| Id: string; Op: 'TOp2; Arg0: ComputationGraph<'TData, 'TOp1, 'TOp2>; Arg1: ComputationGraph<'TData, 'TOp1, 'TOp2> |}

module ComputationGraph =
  let rec cata fArg fOp1 fOp2 (g: ComputationGraph<'TData, 'TOp1, 'TOp2>): 'State =
    let recurse g = cata fArg fOp1 fOp2 g

    match g with
    | Arg a -> fArg a.Id a.TrackGradient
    | Op1 o -> fOp1 o.Id o.Op (recurse o.Arg)
    | Op2 o -> fOp2 o.Id o.Op (recurse o.Arg0) (recurse o.Arg1)

  let fold fArg fOp1 fOp2 (acc: 'State) (g: ComputationGraph<'TData, 'TOp1, 'TOp2>): 'State =
    let rec loop t cont =
      match t with
      | Arg a -> cont (fArg a.Id acc)
      | Op1 o -> loop o.Arg (fun acc ->
                              cont (fOp1 o.Id o.Op acc))
      | Op2 o -> loop o.Arg0 (fun acc0 ->
                              loop o.Arg1 (fun acc1 ->
                              cont (fOp2 o.Id o.Op acc0 acc1)))

    loop g id

  let toString<'TData, 'TOp1, 'TOp2> (g: ComputationGraph<'TData, 'TOp1, 'TOp2>): string =
    let fArg id tg =
      sprintf "%s[TG=%b]" id tg

    let fOp1 id op acc =
      sprintf "%A[%s]( %s )" op id acc

    let fOp2 id op lAcc rAcc =
      sprintf "%A[%s]( %s, %s )" op id lAcc rAcc

    cata fArg fOp1 fOp2 g

  let forward fArg fOp1 fOp2 (g: ComputationGraph<'TData, 'TOperation1, 'TOperation2>) =
    let fArg' id _ =
      let ret = fArg id
      ret, []

    let fOp1' id op (arg, m) =
      let ret = fOp1 op arg
      let m = (id, [| arg |]) :: m
      ret, m

    let fOp2' id op (arg0, m0) (arg1, m1) =
      let ret = fOp2 op arg0 arg1
      let m = m0 @ m1
      let m = (id, [| arg0; arg1 |]) :: m
      ret, m

    let J, cache = cata fArg' fOp1' fOp2' g
    J, cache |> Map.ofList

  let predict fArg fOp1 fOp2 (g: ComputationGraph<'TData, 'TOperation1, 'TOperation2>) =
    let fArg' id _ =
      fArg id

    let fOp1' _ = fOp1

    let fOp2' _ = fOp2

    cata fArg' fOp1' fOp2' g

  let rec backPropagate fOp1 fOp2 (grad0: 'TData) (g: ComputationGraph<'TData, 'TOp1, 'TOp2>): Map<string, 'TData> =
    let recurse = backPropagate fOp1 fOp2

    match g with
    | Arg a ->
      if a.TrackGradient then Map.empty |> Map.add a.Id grad0 else Map.empty
    | Op1 o ->
      let outG = fOp1 grad0 o.Id o.Op
      recurse outG o.Arg
    | Op2 o ->
      let (outG0, outG1) = fOp2 grad0 o.Id o.Op
      let rGradients = recurse outG0 o.Arg0
      let lGradients = recurse outG1 o.Arg1
      rGradients |> Map.fold (fun acc key value -> Map.add key value acc) lGradients
