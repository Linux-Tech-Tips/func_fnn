#include "util.h"

util_err_t util_saveNetwork(network_t * net, char const * filename) {
    /* Opening file and checking success */
    FILE * fp = fopen(filename, "w");
    if(!fp)
	return UTIL_ERR_FILE;

    /* Saving network struct itself */
    fwrite((void *)net, sizeof(network_t), 1, fp);
    /* Saving layer data - groups of (matrix, matrix data, activation) */
    for(size_t idx = 0; idx < net->depth; ++idx) {
	fwrite((void *)(net->weights + idx), sizeof(matrix_t), 1, fp);
	fwrite((void *)(net->weights + idx)->data, sizeof(MATRIX_TYPE), (net->weights + idx)->dataLen, fp);
	fwrite((void *)(&(net->activations + idx)->type), sizeof(activation_type_t), 1, fp);
    }
    return UTIL_OK;
}

util_err_t util_loadNetwork(network_t * net, char const * filename) {
    /* Opening file and checking success */
    FILE * fp = fopen(filename, "r");
    if(!fp)
	return UTIL_ERR_FILE;

    /* Loading network struct */
    if(fread((void *)net, sizeof(network_t), 1, fp) != 1)
	return UTIL_ERR_READ;
    /* Allocating dynamic components of network struct */
    net->weights = (matrix_t *)(malloc(net->depth * sizeof(matrix_t)));
    net->activations = (activation_t *)(malloc(net->depth * sizeof(activation_t)));
    /* Loading layer data */
    for(size_t idx = 0; idx < net->depth; ++idx) {
	/* Reading weights matrix */
	if(fread((void *)(net->weights + idx), sizeof(matrix_t), 1, fp) != 1)
	    return UTIL_ERR_READ;
	/* Allocating space for matrix data and reading matrix data from file */
	(net->weights + idx)->data = (MATRIX_TYPE *)malloc((net->weights + idx)->dataLen * sizeof(MATRIX_TYPE));
	if(fread((void *)(net->weights + idx)->data, sizeof(MATRIX_TYPE), (net->weights + idx)->dataLen, fp) != (net->weights + idx)->dataLen)
	    return UTIL_ERR_READ;
	/* Loading activation function based off file */
	activation_type_t type;
	if(fread((void *)(&type), sizeof(activation_type_t), 1, fp) != 1)
	    return UTIL_ERR_READ;
    }
    return UTIL_OK;
}

util_err_t util_loadPoints(set_t * set, char const * filename) {
    // TODO STUB
    return UTIL_OK;
}

util_err_t util_heatMap(network_t * net, int32_t startPointX, int32_t startPointY, size_t sizeX, size_t sizeY, float step, char * charset, size_t charsetLength) {
    // TODO STUB
    return UTIL_OK;
}
