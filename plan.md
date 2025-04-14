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

