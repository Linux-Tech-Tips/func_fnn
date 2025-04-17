/** 
 * @file util.h
 * @author Linux-Tech-Tips (Martin)
 * @brief File containing NN utility functions
 */
#ifndef UTIL_H
#define UTIL_H

#include <string.h>

#include "matrix.h"
#include "network.h"
#include "activation.h"
#include "set.h"

#define UTIL_POINTS_LOAD_BUFF 1024

/** Error type for utility functions */
typedef enum {
    /** Successful execution state */
    UTIL_OK = 0,
    /** Error opening a file */
    UTIL_ERR_FILE = 1,
    /** Error reading from file */
    UTIL_ERR_READ = 2,
    /** Error with passed function parameters */
    UTIL_ERR_PARAM = 3,
    /** General util error */
    UTIL_ERR = 4
} util_err_t;

/** Network training configuration options for a single hidden layer network */
typedef struct {

    /** Hidden layer size */
    size_t hiddenSize;
    /** Hidden layer activation function type */
    activation_type_t hiddenActivation;
    /** Output layer activation function type */
    activation_type_t outputActivation;

    /** network weightRandMin weight init constant */
    int32_t weightRandMin;
    /** network weightRandMax weight init constant */
    int32_t weightRandMax;
    /** network weightRandDiv weight init constant */
    MATRIX_TYPE weightRandDiv;

    /** Network training learning rate */
    float learningRate;
    /** Network training iteration count */
    size_t itCount;

} util_config_t;

/** Saves an existing network_t data structure to the given file */
util_err_t util_saveNetwork(network_t * net, char const * filename);

/** Loads a saved network from a file into an empty (zero-initialized) network_t structure (don't use network_init) */
util_err_t util_loadNetwork(network_t * net, char const * filename);

/** Loads a dataset of points from a given file into an empty (zero-initialized) set_t structure,
 * in the following format: each line 3 comma-separated floats, representing (xInput,yInput,expectedOutput) */
util_err_t util_loadPoints(set_t * set, char const * filename);

/** Loads a dataset/network configuration file */
util_err_t util_loadConfig(util_config_t * config, char const * filename);

/** Prints a heatmap to standard output of inference of a given network (x,y,1.0)->(z) over the area of the given size from the given start point, using the given charset */
util_err_t util_heatmap(network_t * net, float startPointX, float startPointY, float sizeX, float sizeY, float step, char * charset, size_t charsetLength);

#endif /* UTIL_H */
