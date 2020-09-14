namespace KS.FsDNN

(*
- Assume
  - Binary tree with optional right child
  - No splits

- J = 3 * (a + bc)
  - forward
  - backPropagate
- generalize for matrix
- generalize functions
- track gradient flag


 *)

[<AutoOpen>]
module ComputationGraphDomain =

  type ComputationGraph<'TData, 'TOperation1, 'TOperation2> =
    | Arg of 'TData
    | Operation1 of {| Op: 'TOperation1; Arg: ComputationGraph<'TData, 'TOperation1, 'TOperation2> |}
    | Operation2 of {| Op: 'TOperation2; Arg0: ComputationGraph<'TData, 'TOperation1, 'TOperation2>; Arg1: 'TData |}


module ComputationGraph =

  let rec fold fArg fOperation1 fOperation2 acc (cg: ComputationGraph<'TData, 'TOperation1, 'TOperation2>) : 'TReturn =
    let recurse = fold fArg fOperation1 fOperation2
    match cg with
    | Arg a ->
      fArg acc a
    | Operation1 o ->
      let newAcc = fOperation1 acc o.Op
      recurse newAcc o.Arg
    | Operation2 o ->
      let newAcc = fOperation2 acc o.Op o.Arg1
      recurse newAcc o.Arg0

  let foldBack fArg fOperation1 fOperation2 (cg: ComputationGraph<'TData, 'TOperation1, 'TOperation2>) : 'TReturn =
    let fArg' generator a =
      generator (fArg a)
    let fOperation1' generator op =
      let newGenerator innerValue =
        let newInnerValue = fOperation1 op innerValue
        generator newInnerValue
      newGenerator
    let fOperation2' generator op arg0 =
      let newGenerator innerValue =
        let newInnerValue = fOperation2 op arg0 innerValue
        generator newInnerValue
      newGenerator
    fold fArg' fOperation1' fOperation2' id cg
