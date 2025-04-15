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


typedef void (*activation_func_t) (matrix_t *);

typedef struct {
    activation_func_t f;
    activation_func_t df;
} activation_t;


/** Leaky ReLU elementwise activation function */
void activation_relu_f(matrix_t * m);

/** First derivative of the activation_relu function */
void activation_relu_df(matrix_t * m);

/** Template structure for the relu activation function */
extern activation_t activation_relu;

/** Logistic elementwise activation function */
void activation_logistic_f(matrix_t * m);

/** First derivative of the activation_logistic function */
void activation_logistic_df(matrix_t * m);

/** Template structure for the logistic activation function */
extern activation_t activation_logistic;

#endif /* ACTIVATION_H */
