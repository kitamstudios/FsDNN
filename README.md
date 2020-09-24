# FsDNN - Deep Neural Network Library

## Purpose

- Reimplement material from [deeplearning.ai](https://www.deeplearning.ai/) in F# to gain deeper understanding.
- Build a generic .NET Core DNN library for other projects.

## Library design

- TBD

## Features


### Core

- [x] Written in F#/.NET Core 3 using [Math.NET Numerics on MKL](https://numerics.mathdotnet.com/)
- [x] Componentized - each of the aspects below can be tested and extended on it own
- [x] Entirely Test Driven Developed
- [x] Static Computation Graph
- [x] Tensor abstraction (Uniform API for Matrix/Vector with minimal broadcasting support)
- [x] Minimal transfer learning
- [ ] Gradient checking

### Initializations

- [x] He

### Activations: 

- [x] Linear
- [x] Sigmoid
- [ ] ReLU
- [ ] TanH

### Cost Functions

- [x] Mean squared error
- [x] Binary Cross entropy with logits
- [x] Categorical Cross entropy with logits

### Optimization

- [x] Batch Gradient Descent
- [ ] Stochastic/Mini-Batch Gradient Descent
- [ ] Momentum
- [ ] ADAM

### Regularization

- [ ] L2
- [ ] Dropout

### Demos

- [ ] MNIST
- [ ] ?Multilabel?
- [ ] ?Regression?

## Performance

TBD - compare with numpy

## Future Ideas

- [ ] Implment Tensor functions on GPU
- [ ] Implement CNN and RNN class networks based on the Computation Graph
- [ ] Enable dynamic version of  the Computation Graph 

## References

- [ConvNetJS](https://cs.stanford.edu/people/karpathy/convnetjs/docs.html)
- [Hacker's guide to Neural Networks: Computation Graphs](https://karpathy.github.io/neuralnets/)
- [CS231n Winter 2016: Lecture 4: Backpropagation basics](https://www.youtube.com/watch?v=i94OvYb6noo&t=392)
- [CS231n Winter 2016: Lecture 4: Jacobians](https://www.youtube.com/watch?v=i94OvYb6noo&t=2609)
- [fsharpforfunandprofit - Catamorphisms](https://fsharpforfunandprofit.com/posts/recursive-types-and-folds-3/#container)
- [Introduction to Machine Learning - Neural Network](https://tomaszgolan.github.io/introduction_to_machine_learning/markdown/introduction_to_machine_learning_04_nn/introduction_to_machine_learning_04_nn/)
- [Exhaustive list of loss functions](https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/)

## Near term backlog



o Feature complete
  - Multiclass classification => https://sebastianraschka.com/faq/docs/pytorch-crossentropy.html
  - Multilabel multiclass classification => https://github.com/rasbt/stat479-deep-learning-ss19/blob/master/L08_logistic/L08_logistic_slides.pdf
  - use scikit learn to generate datasets

- Refactoring
  - Align layer definitions with keras (machinelearnignmastery)
    - collapse softmax_logistic & sigmoid_logistic? (https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/)
  - Use cached value 
    - in sigmoid backPropagate
    - in CCE backprop save value for softmax
  - optimize matrix operations in tensor layer
  - Remove Tensor R0: lr should not be a tensor
  - Pull optimization out of trainer

- To understand
  - logit, softmax, sigmoid, BCE, stable BCE, multinomial, binomial

- References
 - https://towardsdatascience.com/cross-entropy-for-classification-d98e7f974451
 - https://levelup.gitconnected.com/killer-combo-softmax-and-cross-entropy-5907442f60ba
 - https://peterroelants.github.io/posts/cross-entropy-logistic/
 - https://deepnotes.io/softmax-crossentropy#cross-entropy-loss
 - https://deepai.org/machine-learning-glossary-and-terms/softmax-layer
