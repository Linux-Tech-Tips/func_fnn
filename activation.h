/** 
 * @file activation.h
 * @author Linux-Tech-Tips (Martin)
 * @brief File containing activation functions for matrices
 */
#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <math.h>

#include "matrix.h"

#define RELU_LEAK 0.01


typedef void (*activation_t) (matrix_t *);


/** Leaky ReLU elementwise activation function */
void activation_relu(matrix_t * m);

/** Logistic elementwise activation function */
void activation_logistic(matrix_t * m);

#endif /* ACTIVATION_H */
