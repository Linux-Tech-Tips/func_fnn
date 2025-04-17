/** 
 * @file network.h
 * @author Linux-Tech-Tips (Martin)
 * @brief File containing Neural Network data structrue and functions
 */
#ifndef NETWORK_H
#define NETWORK_H

#include <stdlib.h>

#include "matrix.h"
#include "activation.h"

/** Minimum value for the weight initialization random integer generator */
extern int32_t network_weightRandMin;
/** Maximum value for the weight initialization random integer generator */
extern int32_t network_weightRandMax;
/** Inverse multiplier for the generated integer for random weight initialization - use this to initialize to floating point numbers */
extern float network_weightRandDiv;

/** Data structure representing a neural network */
typedef struct {

    /** The input layer size */
    size_t inSize;
    /** The output layer size */
    size_t outSize;

    /** The total number of layers (including the output, the minimum number is 1 for a single-layer network) */
    size_t depth;
    /** Array of weights for each layer, represented as a matrix */
    matrix_t * weights;
    /** Array of activation functions for each corresponding layer */
    activation_t * activations;

} network_t;

/** Data structure tracking the values within internal network nodes during inference */
typedef struct {
    /** The total number of tracked layers */
    size_t depth;
    /** The sizes of each individual layer */
    size_t * layers;
    /** The data within the network tracker (first array is layer, second is node at layer, third is before/after activation) */
    MATRIX_TYPE *** layerData;
} network_tracker_t;

/** Network error/result type, returned from network.h functions */
typedef enum {
    /** Default state, successful operation */
    NETWORK_OK = 0,
    /** Error with entered parameters */
    NETWORK_ERR_PARAM = 1,
    /** Error with a null pointer entered */
    NETWORK_ERR_NULL = 2,
    /** Error allocating memory for internal network purposes */
    NETWORK_ERR_ALLOC = 3,
    /** Error accessing an invalid index */
    NETWORK_ERR_IDX = 4,
    /** Error performing inference */
    NETWORK_ERR_INFERENCE = 5,
    /** General/unspecified network error */
    NETWORK_ERR = 6
} network_err_t;


/** Initialize a network_t data structure 
 * @param net pointer to the network structure
 * @param inSize the number of inputs passed to the network
 * @param depth the number of layers in the network (including the output layer)
 * @param layers the node count for each layer in the network, the last being the number of outputs from the network
 * @param activations the activation functions applied to each layer
 */
network_err_t network_init(network_t * net, size_t inSize, size_t depth, size_t * layers, activation_t * activations);

/** Initialize the weights of a network_t data structure to random values */
network_err_t network_initWeights(network_t * net);

/** Internal function for weight initialization, generates random numbers for each given index */
MATRIX_TYPE _network_random(size_t idx);

/** Deallocate data used by a network_t structure */
network_err_t network_destroy(network_t * net);

/** Copies over given weights to a given layer in the network (the provided weights matrix must have the appropriate shape) */
network_err_t network_setWeights(network_t * net, size_t layerIdx, matrix_t * weights);

/** Sets the activation function of a given layer in the network */
network_err_t network_setActivation(network_t * net, size_t layerIdx, activation_t activation);

/** Runs network inference, taking data from the provided input matrix and saving data into the provided output matrix 
 * @param input the matrix containing input values, expected to be a column vector of length 'inSize'
 * @param output the matrix which will contain output values once inference is finished, expected to be a column vector of length 'outSize' (or the size of the last layer)
 */
network_err_t network_inference(network_t * net, matrix_t * input, matrix_t * output);

/** Runs network inference, taking data from the provided input matrix and saving data into the provided output matrix, as well as saving the values at all nodes for training/analysis
 * @param input the matrix containing input values, expected to be a column vector of length 'inSize'
 * @param output the matrix which will contain output values once inference is finished, expected to be a column vector of length 'outSize' (or the size of the last layer)
 * @param nodes tracker object for the internal state of the nodes after inference
 */
network_err_t network_inference_track(network_t * net, matrix_t * input, matrix_t * output, network_tracker_t * nodes);

/** Initializes a network internal node tracker data structure 
 * @param layers the node count for each layer in the network, the last being the number of outputs, as an array
 */
network_err_t network_tracker_init(network_tracker_t * tracker, size_t depth, size_t * layers);

/** Destroys a network internal node tracker data structure */
network_err_t network_tracker_destroy(network_tracker_t * tracker);

#endif /* NETWORK_H */
