/** 
 * @file util.h
 * @author Linux-Tech-Tips (Martin)
 * @brief File containing NN utility functions
 */
#ifndef UTIL_H
#define UTIL_H

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
    /** General util error */
    UTIL_ERR = 3
} util_err_t;


/** Saves an existing network_t data structure to the given file */
util_err_t util_saveNetwork(network_t * net, char const * filename);

/** Loads a saved network from a file into an empty (zero-initialized) network_t structure (don't use network_init) */
util_err_t util_loadNetwork(network_t * net, char const * filename);

/** Loads a dataset of points from a given file into an empty (zero-initialized) set_t structure,
 * in the following format: each line 3 comma-separated floats, representing (xInput,yInput,expectedOutput) */
util_err_t util_loadPoints(set_t * set, char const * filename);

/** Prints a heatmap to standard output of inference of a given network (x,y)->(z) over the area of the given size from the given start point, using the given charset */
util_err_t util_heatMap(network_t * net, int32_t startPointX, int32_t startPointY, size_t sizeX, size_t sizeY, float step, char * charset, size_t charsetLength);

#endif /* UTIL_H */
