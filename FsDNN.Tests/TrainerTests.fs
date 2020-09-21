module KS.FsDNN.Tests.Trainer

open KS.FsDNN
open Xunit
open System.Collections.Generic

(*

TODO
- Binomial LR
- Put functions in the nodes directly
- Use cached value in sigmoid backPropagate

- collapse softmax_logistic & sigmoid_logistic?

- Multinomial LR
- Multiclass classification => https://sebastianraschka.com/faq/docs/pytorch-crossentropy.html
- Multilabel multiclass classification => https://github.com/rasbt/stat479-deep-learning-ss19/blob/master/L08_logistic/L08_logistic_slides.pdf
- Regression

- logit, softmax, sigmoid, BCE, stable BCE, multinomial, binomial


- https://towardsdatascience.com/cross-entropy-for-classification-d98e7f974451
- https://levelup.gitconnected.com/killer-combo-softmax-and-cross-entropy-5907442f60ba
- https://peterroelants.github.io/posts/cross-entropy-logistic/
- https://deepnotes.io/softmax-crossentropy#cross-entropy-loss
- https://deepai.org/


 *)

//[<Fact>]
let ``trainWithGD - logistic regression - OR function`` () =
  let n = Net.makeLayers 1 1.0 { N = 2 } [] (CrossEntropyLossLayer {| Classes = 1 |})

  let X = [ [ 0.; 1.; 0.; 1. ]
            [ 0.; 0.; 1.; 1. ] ] |> Tensor.ofListOfList
  let Y = [ [ 0.; 1.; 1.; 1. ] ] |> Tensor.ofListOfList

  let costs = Dictionary<int, Tensor<double>>()
  let cb = fun e _ J -> if e % 1 = 0 then costs.[e] <- J else ()
  let hp = { HyperParameters.Defaults with Epochs = 1 }

  let n = Trainer.trainWithGD cb n X Y hp

  costs.[0] |> shouldBeEquivalentTo [ [ 0.911103 ] ]
  n.Parameters.["W1"] |> shouldBeEquivalentTo [ [ -0.29620595; -0.46038118 ] ]
  n.Parameters.["b1"] |> shouldBeEquivalentTo [ [ 0.0034253831140481175 ] ]

[<Fact>]
let ``trainWithGD - DNN - XOR function`` () =
  ()
