#include "set.h"

set_err_t set_init(set_t * set, size_t size, size_t inSize, size_t outSize) {
    /* Checking parameters */
    if(!set || size == 0 || inSize == 0 || outSize == 0)
	return SET_ERR_PARAM;

    /* Populating dataset params */
    set->size = size;
    set->inSize = inSize;
    set->outSize = outSize;

    /* Allocating memory for inputs and outputs */
    set->in = (matrix_t *)(malloc(set->size * sizeof(matrix_t)));
    set->out = (matrix_t *)(malloc(set->size * sizeof(matrix_t)));
    for(size_t i = 0; i < set->size; ++i) {
	matrix_init((set->in + i), set->inSize, 1);
	matrix_init((set->out + i), set->outSize, 1);
    }

    return SET_OK;
}

set_err_t set_destroy(set_t * set) {
    /* Checking that the given set is not null */
    if(!set)
	return SET_ERR_PARAM;

    /* Destroying dataset matrices */
    for(size_t i = 0; i < set->size; ++i) {
	matrix_destroy(set->in + i);
	matrix_destroy(set->out + i);
    }

    /* Destroying allocated space for inputs and outputs */
    free(set->in);
    set->in = NULL;
    free(set->out);
    set->out = NULL;

    return SET_OK;
}

set_err_t set_setData(set_t * set, size_t idx, MATRIX_TYPE * inData, MATRIX_TYPE * outData) {
    /* Checking given params */
    if(!set || !inData || !outData)
	return SET_ERR_PARAM;

    /* Checking if index correct */
    if(idx >= set->size)
	return SET_ERR_IDX;

    /* Setting corresponding data */
    for(size_t i = 0; i < set->inSize; ++i) {
	matrix_set((set->in + idx), i, 0, inData[i]);
    }
    for(size_t i = 0; i < set->outSize; ++i) {
	matrix_set((set->out + idx), i, 0, outData[i]);
    }
    return SET_OK;
}

set_err_t set_train_i(set_t * set, network_t * net, network_tracker_t * tracker, matrix_t * out, float learnRate) {
    /* Checking given params */
    if(!set || !net || !tracker || !out || set->inSize != net->inSize || set->outSize != net->outSize)
	return SET_ERR_PARAM;

    matrix_t tmpVal = {0};
    matrix_init(&tmpVal, 1, 1);

    /* Executing inference and correcting weights for every data point in set */
    for(size_t idx = 0; idx < set->size; ++idx) {

	/* Running inference with tracking */
	if(network_inference_track(net, (set->in + idx), out, tracker) != NETWORK_OK)
	    return SET_ERR_TRAIN;

	/* Correcting weights */
	MATRIX_TYPE * prevErrD = (MATRIX_TYPE *)(malloc(tracker->layers[net->depth-1] * sizeof(MATRIX_TYPE)));
	/* Output layer weights */
	for(size_t nodeIdx = 0; nodeIdx < tracker->layers[net->depth-1]; ++nodeIdx) {
	    /* Getting local error */
	    MATRIX_TYPE realOut, dOut, localErr;
	    matrix_get(out, nodeIdx, 0, &realOut);
	    matrix_get((set->out + idx), nodeIdx, 0, &dOut);
	    localErr = dOut - realOut;
	    /* Getting value activated using activation derivative */
	    matrix_set(&tmpVal, 0, 0, tracker->layerData[net->depth-1][nodeIdx][0]);
	    net->activations[net->depth-1].df(&tmpVal);
	    MATRIX_TYPE derivativeVal;
	    matrix_get(&tmpVal, 0, 0, &derivativeVal);
	    /* Getting the output node's error function derivative value */
	    prevErrD[nodeIdx] = localErr * derivativeVal;

	    /* Changing weights to current output node */
	    for(size_t weightIdx = 0; weightIdx < tracker->layers[net->depth-2]; ++weightIdx) {
		MATRIX_TYPE deltaWeight = learnRate * prevErrD[nodeIdx] * tracker->layerData[net->depth-2][weightIdx][1];
		MATRIX_TYPE weight;
		matrix_get((net->weights + net->depth - 1), nodeIdx, weightIdx, &weight);
		matrix_set((net->weights + net->depth - 1), nodeIdx, weightIdx, (weight + deltaWeight));
	    }
	}
	/* All hidden layer weights */
	for(int layerIdx = (net->depth - 2); layerIdx >= 0; --layerIdx) {
	    /* Getting local data for temporary current layer error function derivative values */
	    MATRIX_TYPE * tmpErrD = (MATRIX_TYPE *)(malloc(tracker->layers[layerIdx] * sizeof(MATRIX_TYPE)));
	    /* Going through each node in the current layer */
	    for(size_t nodeIdx = 0; nodeIdx < tracker->layers[layerIdx]; ++nodeIdx) {
		/* Getting value activated using activation derivative */
		matrix_set(&tmpVal, 0, 0, tracker->layerData[layerIdx][nodeIdx][0]);
		net->activations[layerIdx].df(&tmpVal);
		MATRIX_TYPE derivativeVal;
		matrix_get(&tmpVal, 0, 0, &derivativeVal);
		/* Getting sum of weights multiplied by previous error derivative values */
		MATRIX_TYPE prevSum = 0;
		for(size_t k = 0; k < (layerIdx + 1); ++k) {
		    MATRIX_TYPE weight;
		    matrix_get((net->weights + layerIdx + 1), k, nodeIdx, &weight);
		    prevSum += prevErrD[k] * weight;
		}
		/* Getting the current node's error function derivative value */
		tmpErrD[nodeIdx] = derivativeVal * prevSum;

		/* Changing weights to current node */
		size_t weightCount = (layerIdx > 0 ? tracker->layers[layerIdx-1] : net->inSize);
		for(size_t weightIdx = 0; weightIdx < weightCount; ++weightIdx) {
		    MATRIX_TYPE prevVal;
		    if(layerIdx > 0)
			prevVal = tracker->layerData[layerIdx-1][weightIdx][1];
		    else
			matrix_get((set->in + idx), weightIdx, 0, &prevVal);
		    MATRIX_TYPE deltaWeight = learnRate * tmpErrD[nodeIdx] * prevVal;
		    MATRIX_TYPE weight;
		    matrix_get((net->weights + layerIdx), nodeIdx, weightIdx, &weight);
		    matrix_set((net->weights + layerIdx), nodeIdx, weightIdx, (weight + deltaWeight));
		}
	    }
	    /* Changing the previous error derivative value to the current tmp one */
	    free(prevErrD);
	    prevErrD = tmpErrD;
	}

	free(prevErrD);
    }

    matrix_destroy(&tmpVal);

    return SET_OK;
}

set_err_t set_train(set_t * set, network_t * net, size_t * layers, float learnRate, size_t iterations) {

    /* Setting up network tracker */
    network_tracker_t tracker = {0};
    network_tracker_init(&tracker, net->depth, layers);

    /* Setting up output matrix */
    matrix_t out = {0};
    matrix_init(&out, net->outSize, 1);

    /* Running X iterations of training */
    for(size_t itCount = 0; itCount < iterations; ++itCount) {
	set_err_t res = set_train_i(set, net, &tracker, &out, learnRate);
	if(res != SET_OK)
	    return res;
    }

    /* Freeing allocated resources */
    network_tracker_destroy(&tracker);
    matrix_destroy(&out);

    return SET_OK;
}
