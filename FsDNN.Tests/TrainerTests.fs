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
- optimize matrix operations in tensor layer
- Remove Tensor R0
- lr should not be a tensor

- Multinomial LR
- Multiclass classification => https://sebastianraschka.com/faq/docs/pytorch-crossentropy.html
- Multilabel multiclass classification => https://github.com/rasbt/stat479-deep-learning-ss19/blob/master/L08_logistic/L08_logistic_slides.pdf
- Regression

- logit, softmax, sigmoid, BCE, stable BCE, multinomial, binomial


- https://towardsdatascience.com/cross-entropy-for-classification-d98e7f974451
- https://levelup.gitconnected.com/killer-combo-softmax-and-cross-entropy-5907442f60ba
- https://peterroelants.github.io/posts/cross-entropy-logistic/
- https://deepnotes.io/softmax-crossentropy#cross-entropy-loss
- https://deepai.org/machine-learning-glossary-and-terms/softmax-layer
 *)

[<Fact>]
let ``trainWithGD - logistic regression - OR function`` () =
  let n = Net.makeLayers 1 1.0 { N = 2 } [] (CrossEntropyLossLayer {| Classes = 1 |})

  let X = [ [ 0.; 1.; 0.; 1. ]
            [ 0.; 0.; 1.; 1. ] ] |> Tensor.ofListOfList
  let Y = [ [ 0.; 1.; 1.; 1. ] ] |> Tensor.ofListOfList

  let costs = Dictionary<int, Tensor<double>>()
  let cb = fun e _ J -> if e % 1 = 0 then costs.[e] <- J else ()
  let hp = { HyperParameters.Defaults with Epochs = 3 }

  let n = Trainer.trainWithGD cb n X Y hp

  costs.[0] |> shouldBeEquivalentTo [ [ 0.911103 ] ]
  costs.[1] |> shouldBeEquivalentTo [ [ 0.907899 ] ]
  costs.[2] |> shouldBeEquivalentTo [ [ 0.904718 ] ]
  n.Parameters.["W1"] |> shouldBeEquivalentTo [ [ -0.28995256; -0.45392965 ] ]
  n.Parameters.["b1"] |> shouldBeEquivalentTo [ [ 0.010229874835574478 ] ]

  let Y' = X |> Net.predict n

  Y' |> shouldBeEquivalentTo [[0.50255745; 0.43052177; 0.39085974; 0.32439376]]

[<Fact>]
let ``trainWithGD - DNN - XOR function`` () =
  ()
