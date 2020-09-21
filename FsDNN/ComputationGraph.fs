namespace KS.FsDNN

[<AutoOpen>]
module ComputationGraphDomain =

  type Cache<'a> = Map<string, 'a[]>

  type Op1Forward<'a> = 'a -> 'a

  type Op1BackPropagate<'a> = Cache<'a> -> string -> 'a -> 'a

  type Op1Functions<'a> =
    { F: Op1Forward<'a>
      B: Op1BackPropagate<'a> }

  type Op2Forward<'a> = 'a -> 'a -> 'a

  type Op2BackPropagate<'a> = Cache<'a> -> string -> 'a -> 'a * 'a

  type Op2Functions<'a> =
    { F: Op2Forward<'a>
      B: Op2BackPropagate<'a> }

  type ComputationGraph<'TData> =
    | Arg of {| Id: string; TrackGradient: bool |}
    | Op1 of {| Id: string; Functions: Op1Functions<'TData>; Arg: ComputationGraph<'TData>; |}
    | Op2 of {| Id: string; Functions: Op2Functions<'TData>; Arg0: ComputationGraph<'TData>; Arg1: ComputationGraph<'TData> |}

module ComputationGraph =
  let rec cata fArg fOp1 fOp2 (g: ComputationGraph<'TData>): 'State =
    let recurse g = cata fArg fOp1 fOp2 g

    match g with
    | Arg a -> fArg a.Id a.TrackGradient
    | Op1 o -> fOp1 o.Id o.Functions (recurse o.Arg)
    | Op2 o -> fOp2 o.Id o.Functions (recurse o.Arg0) (recurse o.Arg1)

  let fold fArg fOp1 fOp2 (acc: 'State) (g: ComputationGraph<'TData>): 'State =
    let rec loop t cont =
      match t with
      | Arg a -> cont (fArg a.Id acc)
      | Op1 o -> loop o.Arg (fun acc ->
                              cont (fOp1 o.Id acc))
      | Op2 o -> loop o.Arg0 (fun acc0 ->
                              loop o.Arg1 (fun acc1 ->
                              cont (fOp2 o.Id acc0 acc1)))

    loop g id

  let toString<'TData> (g: ComputationGraph<'TData>): string =
    let fArg id tg =
      sprintf "%s[TG=%b]" id tg

    let fOp1 id _ acc =
      sprintf "%s( %s )" id acc

    let fOp2 id _ lAcc rAcc =
      sprintf "%s( %s, %s )" id lAcc rAcc

    cata fArg fOp1 fOp2 g

  let forward fArg (g: ComputationGraph<'TData>) =
    let fArg' id _ =
      let ret = fArg id
      ret, []

    let fOp1' id (f: Op1Functions<_>) (arg, m) =
      let ret = f.F arg
      let m = (id, [| arg |]) :: m
      ret, m

    let fOp2' id (f: Op2Functions<_>) (arg0, m0) (arg1, m1) =
      let ret = f.F arg0 arg1
      let m = m0 @ m1
      let m = (id, [| arg0; arg1 |]) :: m
      ret, m

    let J, cache = cata fArg' fOp1' fOp2' g
    J, cache |> Map.ofList

  let predict fArg (g: ComputationGraph<'TData>) =
    let fArg' id _ =
      fArg id

    let fOp1' _ (f: Op1Functions<_>)  arg =
      f.F arg

    let fOp2' _ (f: Op2Functions<_>)  arg0 arg1 =
      f.F arg0 arg1

    cata fArg' fOp1' fOp2' g

  let rec backPropagate fOp1 fOp2 (grad0: 'TData) (g: ComputationGraph<'TData>): Map<string, 'TData> =
    let recurse = backPropagate fOp1 fOp2

    match g with
    | Arg a ->
      if a.TrackGradient then Map.empty |> Map.add a.Id grad0 else Map.empty
    | Op1 o ->
      let outG = (fOp1 o.Functions.B) o.Id grad0
      recurse outG o.Arg
    | Op2 o ->
      let (outG0, outG1) = (fOp2 o.Functions.B) o.Id grad0
      let rGradients = recurse outG0 o.Arg0
      let lGradients = recurse outG1 o.Arg1
      rGradients |> Map.fold (fun acc key value -> Map.add key value acc) lGradients
