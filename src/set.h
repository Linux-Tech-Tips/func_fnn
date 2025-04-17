/** 
 * @file set.h
 * @author Linux-Tech-Tips (Martin)
 * @brief File containing data structures and functions related to training sets
 */
#ifndef SET_H
#define SET_H

#include "matrix.h"
#include "network.h"

/** Data structure containing a training data set */
typedef struct {

    /** The input data */
    matrix_t * in;
    /** The output data corresponding to the input data */
    matrix_t * out;
    /** The length of the data set */
    size_t size;

    /** The size of the input vector */
    size_t inSize;
    /** The size of the output vector */
    size_t outSize;

} set_t;

/** Set file error types */
typedef enum {
    /** Success state, function executed ok */
    SET_OK = 0,
    /** Error with function parameters */
    SET_ERR_PARAM = 1,
    /** Error with a given index being out of bounds */
    SET_ERR_IDX = 2,
    /** Error during training */
    SET_ERR_TRAIN = 3,
    /** General error */
    SET_ERR = 4
} set_err_t;

/** Initializes a given set_t data structure */
set_err_t set_init(set_t * set, size_t size, size_t inSize, size_t outSize);

/** Destroys a given set_t data structure */
set_err_t set_destroy(set_t * set);

/** Sets the given input and output data at a given data point */
set_err_t set_setData(set_t * set, size_t idx, MATRIX_TYPE * inData, MATRIX_TYPE * outData);

/** Trains a single iteration of the given network structure on the given set */
set_err_t set_train_i(set_t * set, network_t * net, network_tracker_t * tracker, matrix_t * out, float learnRate);

/** Trains a given network on a given set, with the given learnRate, for the given number of iterations */
set_err_t set_train(set_t * set, network_t * net, size_t * layers, float learnRate, size_t iterations);

#endif /* TRAIN_H */
