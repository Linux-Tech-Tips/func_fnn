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
	*(net->activations + idx) = activation_get(type);
    }
    return UTIL_OK;
}

util_err_t util_loadPoints(set_t * set, char const * filename) {
    /* Opening file and checking success */
    FILE * fp = fopen(filename, "r");
    if(!fp)
	return UTIL_ERR_FILE;

    /* Loading data line by line */
    char * line = NULL;
    size_t lineSize = 0;
    size_t maxPointsLen = UTIL_POINTS_LOAD_BUFF;
    MATRIX_TYPE ** points = (MATRIX_TYPE **)malloc(sizeof(MATRIX_TYPE *) * maxPointsLen);
    size_t pointsLen = 0;
    while(getline(&line, &lineSize, fp) >= 0) {
	if(lineSize > 0) {
	    MATRIX_TYPE in1 = 0, in2 = 0, out = 0;
	    if(sscanf(line, MATRIX_TYPE_SCANF "," MATRIX_TYPE_SCANF "," MATRIX_TYPE_SCANF, &in1, &in2, &out) == 3) {
		/* Reallocate points array if end of available space reached */
		if(pointsLen == maxPointsLen) {
		    maxPointsLen *= 2;
		    points = (MATRIX_TYPE **)realloc(points, sizeof(MATRIX_TYPE *) * maxPointsLen);
		}
		/* Write processed numbers from file into points array */
		points[pointsLen] = (MATRIX_TYPE *)malloc(sizeof(MATRIX_TYPE) * 4);
		points[pointsLen][0] = in1;
		points[pointsLen][1] = in2;
		points[pointsLen][2] = 1.0f;
		points[pointsLen][3] = out;
		++pointsLen;
	    }
	}
    }
    free(line);
    /* Initialising set_t structure and populating with the read values */
    set_init(set, pointsLen, 3, 1);
    for(size_t idx = 0; idx < pointsLen; ++idx) {
	set_setData(set, idx, points[idx], (points[idx] + 3));
	free(points[idx]);
    }
    free(points);
    return UTIL_OK;
}

util_err_t util_loadConfig(util_config_t * config, char const * filename) {
    if(!config || !filename)
	return UTIL_ERR_PARAM;

    /* Loading file and checking success */
    FILE * fp = fopen(filename, "r");
    if(!fp)
	return UTIL_ERR_FILE;

    /* Loading config file */
    char * line = NULL;
    size_t lineLen = 0;
    while(getline(&line, &lineLen, fp) >= 0) {
	/* Skipping commented lines */
	if(line[0] == '#')
	    continue;
	/* Processing options */
	if(strstr(line, "learning_rate")) {
	    sscanf(line, "learning_rate %f", &config->learningRate);
	} else if(strstr(line, "iteration_count")) {
	    sscanf(line, "iteration_count %lu", &config->itCount);
	} else if(strstr(line, "hidden_size")) {
	    sscanf(line, "hidden_size %lu", &config->hiddenSize);
	} else if(strstr(line, "hidden_activation")) {
	    sscanf(line, "hidden_activation %d", (int *)(&config->hiddenActivation));
	} else if(strstr(line, "output_activation")) {
	    sscanf(line, "output_activation %d", (int *)(&config->outputActivation));
	} else if(strstr(line, "random_int_min")) {
	    sscanf(line, "random_int_min %d", &config->weightRandMin);
	} else if(strstr(line, "random_int_max")) {
	    sscanf(line, "random_int_max %d", &config->weightRandMax);
	} else if(strstr(line, "div_const")) {
	    sscanf(line, "div_const " MATRIX_TYPE_SCANF, &config->weightRandDiv);
	}
    }
    free(line);
    return UTIL_OK;
}

util_err_t util_heatmap(network_t * net, float startPointX, float startPointY, float sizeX, float sizeY, float step, char * charset, size_t charsetLength) {
    if(!net || !charset)
	return UTIL_ERR_PARAM;

    /* Running network inference to generate heatmap values, keeping track of max and min */
    MATRIX_TYPE min, max;
    size_t xLength = (size_t)(sizeX / step);
    size_t yLength = (size_t)(sizeY / step);
    MATRIX_TYPE map [yLength][xLength];
    matrix_t out = {0}, in = {0};
    matrix_init(&out, 1, 1);
    matrix_init(&in, 3, 1);
    for(size_t y = 0; y < yLength; ++y) {
	for(size_t x = 0; x < xLength; ++x) {
	    /* Setting up and performing inference for the current data point */
	    matrix_set(&in, 0, 0, (startPointX + x * step));
	    matrix_set(&in, 1, 0, (startPointY + y * step));
	    matrix_set(&in, 2, 0, 1.0f);
	    network_inference(net, &in, &out);
	    MATRIX_TYPE outVal;
	    matrix_get(&out, 0, 0, &outVal);
	    /* Saving obtained value, updating min, max */
	    map[y][x] = outVal;
	    if(x == 0 && y == 0) {
		min = outVal;
		max = outVal;
	    }
	    if(outVal < min)
		min = outVal;
	    if(outVal > max)
		max = outVal;
	}
    }
    matrix_destroy(&in);
    matrix_destroy(&out);

    /* Printing heatmap based on charset */
    float charStep = (max - min) / (float)(charsetLength - 1);
    for(int32_t y = (yLength - 1); y >= 0; --y) {
	/* Printing left border with numbers */
	if(y == (yLength - 1)) {
	    if(startPointY < 0 && (startPointY + sizeY) >= 0)
		putchar(' ');
	    printf("%.2f | ", (startPointY + sizeY));
	} else {
	    if(startPointY < 0)
		putchar(' ');
	    fputs("     | ", stdout);
	}
	/* Printing line of numbers */
	for(size_t x = 0; x < xLength; ++x) {
	    size_t charIdx = (size_t)((map[y][x] - min) / charStep);
	    printf("%c ", charset[charIdx]);
	}
	putchar('\n');
    }
    /* Printing bottom lines with bar and numbers */
    printf("%.2f \\", startPointY);
    for(size_t x = 0; x < xLength; ++x)
	fputs("--", stdout);
    if(startPointY < 0)
	putchar(' ');
    printf("\n      %.2f", startPointX);
    for(size_t x = 0; x < (xLength - 4); ++x)
	fputs("  ", stdout);
    printf("%.2f\n", (startPointX + sizeX));

    return UTIL_OK;
}
