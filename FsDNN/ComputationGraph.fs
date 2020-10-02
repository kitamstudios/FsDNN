namespace KS.FsDNN

[<AutoOpen>]
module ComputationGraphDomain =

  /// Convention is [| result; arg0; arg1; ... |]
  type CacheValue<'a> = 'a[]

  type Cache<'a> = Map<string, CacheValue<'a>>

  type Op1Forward<'a> = 'a -> CacheValue<'a>

  type Op1BackPropagate<'a> = Cache<'a> -> string -> 'a -> 'a

  type Op1Functions<'a> =
    { F: Op1Forward<'a>
      B: Op1BackPropagate<'a> }

  type Op2Forward<'a> = 'a -> 'a -> CacheValue<'a>

  type Op2BackPropagate<'a> = Cache<'a> -> string -> 'a -> 'a * 'a

  type Op2Functions<'a> =
    { F: Op2Forward<'a>
      B: Op2BackPropagate<'a> }

  type Operation1Definition<'a> =
    { Name: string
      Functions: Op1Functions<'a> }

  type Operation2Definition<'a> =
    { Name: string
      Functions: Op2Functions<'a> }

  type ComputationGraph<'TData> =
    | Arg of {| Id: string; TrackGradient: bool |}
    | Op1 of {| D: Operation1Definition<'TData>; Arg: ComputationGraph<'TData>; |}
    | Op2 of {| D: Operation2Definition<'TData>; Arg0: ComputationGraph<'TData>; Arg1: ComputationGraph<'TData> |}

module ComputationGraph =
  let rec cata fArg fOp1 fOp2 (g: ComputationGraph<'TData>): 'State =
    let recurse g = cata fArg fOp1 fOp2 g

    match g with
    | Arg a -> fArg a.Id a.TrackGradient
    | Op1 o -> fOp1 o.D (recurse o.Arg)
    | Op2 o -> fOp2 o.D (recurse o.Arg0) (recurse o.Arg1)

  let fold fArg fOp1 fOp2 (acc: 'State) (g: ComputationGraph<'TData>): 'State =
    let rec loop t cont =
      match t with
      | Arg a -> cont (fArg a.Id a.TrackGradient acc)
      | Op1 o -> loop o.Arg (fun acc ->
                              cont (fOp1 o.D.Name acc))
      | Op2 o -> loop o.Arg0 (fun acc0 ->
                              loop o.Arg1 (fun acc1 ->
                              cont (fOp2 o.D.Name acc0 acc1)))

    loop g id

  let toString<'TData> (g: ComputationGraph<'TData>): string =
    let fArg id tg =
      sprintf "%s[TG=%b]" id tg

    let fOp1 (d: Operation1Definition<_>) acc =
      sprintf "%s( %s )" d.Name acc

    let fOp2 (d: Operation2Definition<_>) lAcc rAcc =
      sprintf "%s( %s, %s )" d.Name lAcc rAcc

    cata fArg fOp1 fOp2 g

  let forward fArg (g: ComputationGraph<'TData>) =
    let fArg' id _ =
      let ret = fArg id
      ret, []

    let fOp1' (d: Operation1Definition<_>) (arg, m) =
      let cv = d.Functions.F arg
      let m = (d.Name, cv) :: m
      cv.[0], m

    let fOp2' (d: Operation2Definition<_>) (arg0, m0) (arg1, m1) =
      let cv = d.Functions.F arg0 arg1
      let m = m0 @ m1
      let m = (d.Name, cv) :: m
      cv.[0], m

    let J, cache = cata fArg' fOp1' fOp2' g
    J, (cache |> Map.ofList)

  let predict fArg (g: ComputationGraph<'TData>) =
    let fArg' id _ =
      fArg id

    let fOp1' (d: Operation1Definition<_>) arg =
      d.Functions.F arg |> Array.head

    let fOp2' (d: Operation2Definition<_>) arg0 arg1 =
      d.Functions.F arg0 arg1 |> Array.head

    cata fArg' fOp1' fOp2' g

  let rec backPropagate fOp1 fOp2 (grad0: 'TData) (g: ComputationGraph<'TData>): Map<string, 'TData> =
    let recurse = backPropagate fOp1 fOp2

    match g with
    | Arg a ->
      if a.TrackGradient then Map.empty |> Map.add a.Id grad0 else Map.empty
    | Op1 o ->
      let outG = (fOp1 o.D.Functions.B) o.D.Name grad0
      recurse outG o.Arg
    | Op2 o ->
      let (outG0, outG1) = (fOp2 o.D.Functions.B) o.D.Name grad0
      let rGradients = recurse outG0 o.Arg0
      let lGradients = recurse outG1 o.Arg1
      rGradients |> Map.fold (fun acc key value -> Map.add key value acc) lGradients
