# Neural Network

Basically, any layer can be represented as a Matrix, and inference is straight up just a Matrix multiplication.

```
| w11 w12 w12 |   | in1 |   | in1*w11 + in2*w12 + in3*w13 |   | w1 * in |
| w21 w22 w23 | * | in2 | = | in1*w21 + in2*w22 + in3*w23 | = | w2 * in |
| w31 w32 w33 |   | in3 |   | in1*w31 + in2*w32 + in3*w33 |   | w3 * in |
```

After this, we can run the output vector through an activation function, which will give us the desired result.

Meaning, inference is literally a few matrix products with activation functions in em.

The plan is therefore:

- Make Matrix class, have vectors be Nx1 matrices
- Then, the entire network is basically a list of matrix-activation pairs, and inference is:
  - for each matrix in the list, multiply M * LastIn
  - Run LastIn through activation function, keep it for the next iteration
  - Once no more matrices left, what's in LastIn is the result
- Training will be somewhat more difficult xd

## Notes about network structure

```
layers: In -> W1 -> W2 -> Out
            -weights W1\*In
                  -weights W2\*W1
                        -weights Out\*W2
```


## Plan for network training

Given training data, a set of input/output matrix pairs, we can use gradient descent to train the perceptron

Necessary variables:

- Inputs (X matrices with the same shape as the network input)
- Outputs (X matrices with the same shape as the network output)
- Learning rate (hyperparameter floating point number affecting the weights convergence)
- Derivative of any used activation function

Internal representation:

- Struct with matrix lists, length of training data and learning rate
- Function running training for N iterations using a training structure

Required stuff:

- For the change of a weight to an output node, we need:
  - value of node *i* in previous layer connected to the weight
  - value of current output node before activation
  - derivative of the current activation
  - error at current output node (difference between the expected value and inferred value)
  - learning rate
- Part of this gives us the derivative of the error function based on the current node inputs
- For the change of a weight to a hidden node, we need:
  - value of node *i* in previous layer connected to the weight
  - value of current node before activation
  - derivative of the current activation
  - all current weights leading to the next layer from the current node
  - all derivatives of the error function based on the node inputs for the nodes in the next layer
  - learning rate

In code, this translates to keeping track of:

- all weights of the network
- all values of each node, before and after activation, in the network, after inference
- activations and their derivatives for each layer
- expected value and inferred value at each output
- calculated error function derivatives for each node in the previously calculated layer, for anything after output

This will require the inference function to be able to keep track of the calculated values at nodes.
