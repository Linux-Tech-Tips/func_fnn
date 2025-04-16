#include "network.h"

network_err_t network_init(network_t * net, size_t inSize, size_t depth, size_t * layers, activation_t * activations) {
    /* Checking validity of properties */
    if(inSize < 1 || depth < 1)
	return NETWORK_ERR_PARAM;
    if(!net || !layers || !activations)
	return NETWORK_ERR_NULL;

    /* Assigning properties */
    net->inSize = inSize;
    net->outSize = layers[depth - 1];
    net->depth = depth;

    /* Allocating weights and activations size */
    net->weights = (matrix_t *)(malloc(net->depth * sizeof(matrix_t)));
    net->activations = (activation_t *)(malloc(net->depth * sizeof(activation_t)));

    /* Checking weights allocation */
    if(!net->weights || !net->activations)
	return NETWORK_ERR_ALLOC;

    /* Initialising weights and activations */
    for(size_t i = 0; i < depth; ++i) {
	/* Allocate matrices for the weights */
	if(matrix_init((net->weights + i), layers[i], (i > 0 ? layers[i-1] : inSize)) != MATRIX_OK)
	    return NETWORK_ERR_ALLOC;
	/* Copy over activation functions */
	net->activations[i] = activations[i];
    }
    return NETWORK_OK;
}

network_err_t network_destroy(network_t * net) {
    /* Destroying associated weights */
    for(size_t i = 0; i > net->depth; ++i) {
	matrix_destroy((net->weights + i));
    }
    /* Freeing space allocated for weights and activations */
    free(net->weights);
    free(net->activations);
    return NETWORK_OK;
}

network_err_t network_setWeights(network_t * net, size_t layerIdx, matrix_t * weights) {
    /* Validating arguments */
    if(!net || !weights)
	return NETWORK_ERR_NULL;
    if(layerIdx >= net->depth)
	return NETWORK_ERR_IDX;
    /* Copying over weights */
    if(matrix_copy(weights, (net->weights + layerIdx)) != MATRIX_OK)
	return NETWORK_ERR;
    return NETWORK_OK;
}

network_err_t network_setActivation(network_t * net, size_t layerIdx, activation_t activation) {
    /* Validating arguments */
    if(!net)
	return NETWORK_ERR_NULL;
    if(layerIdx >= net->depth)
	return NETWORK_ERR_IDX;
    /* Copying over activation function */
    net->activations[layerIdx] = activation;
    return NETWORK_OK;
}

network_err_t network_inference(network_t * net, matrix_t * input, matrix_t * output) {
    return network_inference_track(net, input, output, NULL);
}

network_err_t network_inference_track(network_t * net, matrix_t * input, matrix_t * output, network_tracker_t * nodes) {
    /* Validating arguments */
    if(!net || !input || !output)
	return NETWORK_ERR_NULL;
    if(input->rows != net->inSize || output->rows != net->outSize)
	return NETWORK_ERR_PARAM;

    /* Running inference (a series of matrix multiplication) */
    matrix_t prevResult = {0};
    if(matrix_init(&prevResult, net->inSize, 1) != MATRIX_OK)
	return NETWORK_ERR_ALLOC;
    if(matrix_copy(input, &prevResult) != MATRIX_OK)
	return NETWORK_ERR_INFERENCE;

    /* Iteratively performing matrix multiplication */
    for(size_t layerIdx = 0; layerIdx < net->depth; ++layerIdx) {

	/* Allocate space for temporary result */
	matrix_t tmpResult = {0};
	if(matrix_init(&tmpResult, (net->weights + layerIdx)->rows, 1) != MATRIX_OK)
	    return NETWORK_ERR_ALLOC;

	/* Do matrix multiplication */
	if(matrix_matmul((net->weights + layerIdx), &prevResult, &tmpResult) != MATRIX_OK)
	    return NETWORK_ERR_INFERENCE;
	/* Free temporarily allocated space, copy temporary result over to the previous result */
	if(matrix_destroy(&prevResult) != MATRIX_OK)
	    return NETWORK_ERR_ALLOC;

	/* Attempting to write to nodes before activation */
	if(nodes) {
	    for(size_t i = 0; i < tmpResult.rows; ++i) {
		MATRIX_TYPE num;
		matrix_get(&tmpResult, i, 0, &num);
		nodes->layerData[layerIdx][i][0] = num;
	    }
	}

	/* Calling activation function */
	net->activations[layerIdx].f(&tmpResult);

	/* Attempting to write nodes after activation */
	if(nodes) {
	    for(size_t i = 0; i < tmpResult.rows; ++i) {
		MATRIX_TYPE num;
		matrix_get(&tmpResult, i, 0, &num);
		nodes->layerData[layerIdx][i][1] = num;
	    }
	}

	/* Overwriting the previous result with the current temporary result */
	prevResult = tmpResult;
    }

    /* Writing data to output and freeing allocated resources */
    if(matrix_copy(&prevResult, output) != MATRIX_OK)
	return NETWORK_ERR_INFERENCE;
    if(matrix_destroy(&prevResult) != MATRIX_OK)
	return NETWORK_ERR_ALLOC;
    return NETWORK_OK;
}

network_err_t network_tracker_init(network_tracker_t * tracker, size_t depth, size_t * layers) {
    tracker->depth = depth;
    tracker->layers = (size_t *)(malloc(depth * sizeof(size_t)));
    tracker->layerData = (MATRIX_TYPE ***)(malloc(tracker->depth * sizeof(MATRIX_TYPE **)));
    for(size_t i = 0; i < depth; ++i) {
	tracker->layers[i] = layers[i];
	tracker->layerData[i] = (MATRIX_TYPE **)(malloc(tracker->layers[i] * sizeof(MATRIX_TYPE *)));
	for(size_t j = 0; j < tracker->layers[i]; ++j) {
	    tracker->layerData[i][j] = (MATRIX_TYPE *)(malloc(2 * sizeof(MATRIX_TYPE)));
	}
    }
    return NETWORK_OK;
}

network_err_t network_tracker_destroy(network_tracker_t * tracker) {
    for(size_t i = 0; i < tracker->depth; ++i) {
	for(size_t j = 0; j < tracker->layers[i]; ++j) {
	    free(tracker->layerData[i][j]);
	}
	free(tracker->layerData[i]);
    }
    free(tracker->layerData);
    free(tracker->layers);
    return NETWORK_OK;
}
