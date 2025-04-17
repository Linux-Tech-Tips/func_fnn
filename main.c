/** 
 * @file main.h
 * @author Linux-Tech-Tips (Martin)
 * @brief Main file, containing project runtime
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#include "matrix.h"
#include "activation.h"
#include "network.h"
#include "set.h"
#include "util.h"

#ifndef MAIN_NETWORK_FILENAME
#define MAIN_NETWORK_FILENAME "active.net"
#endif /* NETWORK_FILENAME */

#ifndef MAIN_HEATMAP_ORIGIN_X
#define MAIN_HEATMAP_ORIGIN_X -2
#endif /* MAIN_HEATMAP_ORIGIN_X */
#ifndef MAIN_HEATMAP_ORIGIN_Y
#define MAIN_HEATMAP_ORIGIN_Y -1
#endif /* MAIN_HEATMAP_ORIGIN_Y */

#ifndef MAIN_HEATMAP_SIZE_X
#define MAIN_HEATMAP_SIZE_X 4
#endif /* MAIN_HEATMAP_SIZE_X */
#ifndef MAIN_HEATMAP_SIZE_Y
#define MAIN_HEATMAP_SIZE_Y 3
#endif /* MAIN_HEATMAP_SIZE_Y */

#ifndef MAIN_HEATMAP_STEP
#define MAIN_HEATMAP_STEP 0.1f
#endif /* MAIN_HEATMAP_STEP */

void main_printHelp(char * programName);

short main_loadNet(network_t * net, char const * networkFile);

void main_train(char const * pointsFile, char const * configFile);

void main_point(MATRIX_TYPE x, MATRIX_TYPE y);

void main_heatmap(float originX, float originY, float sizeX, float sizeY, float step);

int main(int argc, char ** argv) {

    /* Initialising random number generator */
    srand(time(NULL));

    /* Checking command line arguments */
    if(argc < 2) {
	printf("Usage: '%s <command> <options>'\nTry '%s help'\n", argv[0], argv[0]);
	return 1;
    }

    /* Process command */
    if(strcmp(argv[1], "help") == 0 || strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0) {
	main_printHelp(argv[0]);
	return 0;

    } else if(strcmp(argv[1], "train") == 0) {
	if(argc < 4) {
	    printf("Error: not enough arguments for 'train' command\nTry '%s help'\n", argv[0]);
	    return 1;
	} else {
	    main_train(argv[2], argv[3]);
	}
	
    } else if(strcmp(argv[1], "point") == 0) {
	if(argc < 4) {
	    printf("Error: not enough arguments for 'point' command\nTry '%s help'\n", argv[0]);
	    return 1;
	} else {
	    MATRIX_TYPE x = 0, y = 0;
	    sscanf(argv[2], MATRIX_TYPE_SCANF, &x);
	    sscanf(argv[3], MATRIX_TYPE_SCANF, &y);
	    main_point(x, y);
	}

    } else if(strcmp(argv[1], "heatmap") == 0) {
	float originX = MAIN_HEATMAP_ORIGIN_X;
	if(argc > 2)
	    sscanf(argv[2], "%f", &originX);
	float originY = MAIN_HEATMAP_ORIGIN_Y;
	if(argc > 3)
	    sscanf(argv[3], "%f", &originY);
	float sizeX = MAIN_HEATMAP_SIZE_X;
	if(argc > 4)
	    sscanf(argv[4], "%f", &sizeX);
	float sizeY = MAIN_HEATMAP_SIZE_Y;
	if(argc > 5)
	    sscanf(argv[5], "%f", &sizeY);
	float step = MAIN_HEATMAP_STEP;
	if(argc > 6)
	    sscanf(argv[6], "%f", &step);
	main_heatmap(originX, originY, sizeX, sizeY, step);

    } else {
	printf("Error: unrecognized command '%s'\nTry '%s help'\n", argv[1], argv[0]);
	return 1;
    }

    return 0;
}

void main_printHelp(char * programName) {
    printf("Usage: '%s <command> <options>'\n", programName);
    puts("available <command>s and their <options>:\n"
	 "  - train <points> <config> ............ train neural network with given points and config files\n"
	 "  - point <x> <y> ...................... run inference and provide an output value for a given point (x,y)\n"
	 "  - heatmap [origin_x] [origin_y]\n"
	 "            [size_x] [size_y] [step] ... run inference (optionally specify a custom area of size (size_x,size_y) from origin) and display heatmap\n"
	 "  - --help | -h | help ................. display this help menu");
}

short main_loadNet(network_t * net, char const * networkFile) {
    if(util_loadNetwork(net, networkFile) != UTIL_OK) {
	printf("Error: Network could not be loaded\nCheck if file '%s' exists?\n", networkFile);
	return 0;
    }
    return 1;
}

void main_train(char const * pointsFile, char const * configFile) {
    /* Load config file */
    util_config_t conf = {0};
    if(util_loadConfig(&conf, configFile) != UTIL_OK) {
	printf("Error: Config coould not be loaded\nCheck if file '%s' exists?\n", configFile);
	return;
    }

    /* Load points file */
    set_t set = {0};
    if(util_loadPoints(&set, pointsFile) != UTIL_OK) {
	printf("Error: Training points coould not be loaded\nCheck if file '%s' exists?\n", pointsFile);
	return;
    }

    /* Initialize network */
    network_t net = {0};
    size_t layers [2] = { conf.hiddenSize, 1 };
    activation_t activations [2] = { activation_get(conf.hiddenActivation), activation_get(conf.outputActivation) };
    network_init(&net, 3, 2, layers, activations);
    /* Set weights configs and initialize weights */
    network_weightRandMin = conf.weightRandMin;
    network_weightRandMax = conf.weightRandMax;
    network_weightRandDiv = conf.weightRandDiv;
    network_initWeights(&net);

    /* Run training */
    set_train(&set, &net, layers, conf.learningRate, conf.itCount);

    /* Save network */
    util_saveNetwork(&net, MAIN_NETWORK_FILENAME);

    /* Dispose of any allocated/initialised resources */
    set_destroy(&set);
    network_destroy(&net);
}

void main_point(MATRIX_TYPE x, MATRIX_TYPE y) {
    /* Load network */
    network_t net = {0};
    main_loadNet(&net, MAIN_NETWORK_FILENAME);

    /* Set up and run inference */
    matrix_t in = {0}, out = {0};
    matrix_init(&in, 3, 1);
    matrix_init(&out, 1, 1);
    matrix_set(&in, 0, 0, x);
    matrix_set(&in, 1, 0, y);
    matrix_set(&in, 2, 0, 1.0f);
    network_inference(&net, &in, &out);

    /* Print result */
    MATRIX_TYPE outVal = 0;
    matrix_get(&out, 0, 0, &outVal);
    printf("Inference result: (" MATRIX_TYPE_PRINTF ", " MATRIX_TYPE_PRINTF ") -> (" MATRIX_TYPE_PRINTF ")\n", x, y, outVal);

    /* Dispose of any allocated resources */
    matrix_destroy(&in);
    matrix_destroy(&out);
    network_destroy(&net);
}

void main_heatmap(float originX, float originY, float sizeX, float sizeY, float step) {
    /* Load network */
    network_t net = {0};
    main_loadNet(&net, MAIN_NETWORK_FILENAME);

    /* Generate heatmap */
    char charset [7] = {'.', ',', '-', ';', '!', 'I', 'H'};
    util_heatmap(&net, originX, originY, sizeX, sizeY, step, charset, 7);

    /* Dispose of any allocated resources */
    network_destroy(&net);
}
