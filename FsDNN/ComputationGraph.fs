namespace KS.FsDNN

(*
- Assume
  - Binary tree with optional right child
  - No splits

- J = ((x + y) * z) ^ 2
  - forward
  - backPropagate
- generalize for matrix
- generalize functions
- track gradient flag


 *)

[<AutoOpen>]
module ComputationGraphDomain =

  type ComputationGraph<'TData, 'TOp1, 'TOp2> =
    | Arg of Value<'TData>
    | Op1 of Op1Info<'TData, 'TOp1, 'TOp2>
    | Op2 of Op2Info<'TData, 'TOp1, 'TOp2>
  and Value<'TData> = { mutable Data: 'TData; mutable Gradient: 'TData }
  and Op1Info<'TData, 'TOp1, 'TOp2> = { Op: 'TOp1; mutable IVal: Value<'TData>; mutable In: 'TData; Arg: ComputationGraph<'TData, 'TOp1, 'TOp2>; }
  and Op2Info<'TData, 'TOp1, 'TOp2> = { Op: 'TOp2; mutable IVal: Value<'TData>; mutable In0: 'TData; mutable In1: 'TData; Arg0: Value<'TData>; Arg1: ComputationGraph<'TData, 'TOp1, 'TOp2> }

module ComputationGraph =

  let rec cata fArg fOp1 fOp2 (g: ComputationGraph<'TData, 'TOperation1, 'TOperation2>) : 'TData =
    let recurse = cata fArg fOp1 fOp2
    match g with
    | Arg a ->
      fArg a
    | Op1 o ->
      fOp1 o (recurse o.Arg)
    | Op2 o ->
      fOp2 o (recurse o.Arg1)

  let rec fold fArg fOp1 fOp2 acc (g: ComputationGraph<'TData, 'TOperation1, 'TOperation2>) : 'TReturn =
    let recurse = fold fArg fOp1 fOp2
    match g with
    | Arg a ->
      fArg acc a
    | Op1 o ->
      let newAcc = fOp1 acc o
      recurse newAcc o.Arg
    | Op2 o ->
      let newAcc = fOp2 acc o
      recurse newAcc o.Arg1

  let forward fArg fOp1 fOp2 (g: ComputationGraph<'TData, 'TOperation1, 'TOperation2>) =
    let fArg' = fArg

    let fOp1' (o: Op1Info<_, _, _>) arg =
      let x = fOp1 o.Op arg
      o.IVal.Data <- x
      o.In <- arg
      x

    let fOp2' o arg1 =
      let x = fOp2 o.Op o.Arg0.Data arg1
      o.IVal.Data <- x
      o.In0 <- o.Arg0.Data
      o.In1 <- arg1
      x

    cata fArg' fOp1' fOp2' g

  let rec backPropagate fArg fOp1 fOp2 acc (g: ComputationGraph<'TData, 'TOperation1, 'TOperation2>) : 'TReturn =
    let recurse = backPropagate fArg fOp1 fOp2

    match g with
    | Arg a ->
      fArg acc a
    | Op1 o ->
      let acc' = fOp1 acc o
      recurse acc' o.Arg
    | Op2 o ->
      let acc' = fOp2 acc o
      recurse acc' o.Arg1
