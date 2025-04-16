/* Main file, testing for now */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "matrix.h"
#include "activation.h"
#include "network.h"
#include "set.h"

int main(void) {

    srand(time(NULL));

    matrix_t m1 = {0};
    matrix_init(&m1, 2, 3);

    matrix_t m2 = {0};
    matrix_init(&m2, 3, 2);

    matrix_set(&m1, 0, 0, 1);
    matrix_set(&m1, 0, 1, 2);
    matrix_set(&m1, 0, 2, 3);
    matrix_set(&m1, 1, 0, 7);
    matrix_set(&m1, 1, 1, 8);
    matrix_set(&m1, 1, 2, 9);

    matrix_set(&m2, 0, 0, 5);
    matrix_set(&m2, 0, 1, 8);
    matrix_set(&m2, 1, 0, 6);
    matrix_set(&m2, 1, 1, -7);
    matrix_set(&m2, 2, 0, 2);
    matrix_set(&m2, 2, 1, 1);

    /* Testing matrix multiplication */
    matrix_t result = {0};
    matrix_init(&result, 2, 2);

    matrix_err_t mul = matrix_matmul(&m1, &m2, &result);

    if(mul != MATRIX_OK) {
	puts("Matrix Error!");
	return 1;
    }

    matrix_print(&result);

    activation_t a1 = activation_relu;
    activation_t a2 = activation_logistic;

    a1.f(&result);
    matrix_print(&result);

    a2.f(&result);
    matrix_print(&result);

    matrix_destroy(&m1);
    matrix_destroy(&m2);

    /* Testing network */

    network_t testNet = {0};
    size_t layers [] = {4, 1};
    activation_t activations [] = {activation_logistic, activation_logistic};
    network_init(&testNet, 3, 2, layers, activations);

    matrix_t weights = {0};
    matrix_init(&weights, 4, 3);
    matrix_set(&weights, 0, 0, 0.8f); matrix_set(&weights, 0, 1, -1.1f); matrix_set(&weights, 0, 2, 1.0f);
    matrix_set(&weights, 1, 0, -1.3f); matrix_set(&weights, 1, 1, -2.7f); matrix_set(&weights, 1, 2, 4.4f);
    matrix_set(&weights, 2, 0, 0.2f); matrix_set(&weights, 2, 1, 2.1f); matrix_set(&weights, 2, 2, -1.1f);
    matrix_set(&weights, 3, 0, 0.0f); matrix_set(&weights, 3, 1, 0.0f); matrix_set(&weights, 3, 2, 1.0f);
    network_setWeights(&testNet, 0, &weights);
    matrix_destroy(&weights);

    matrix_init(&weights, 1, 4);
    matrix_set(&weights, 0, 0, 1);
    matrix_set(&weights, 0, 1, 1);
    matrix_set(&weights, 0, 2, 1);
    matrix_set(&weights, 0, 3, -2.75);
    network_setWeights(&testNet, 1, &weights);
    matrix_destroy(&weights);

    network_setActivation(&testNet, 1, activation_relu);


    matrix_t in = {0};
    matrix_init(&in, 3, 1);
    matrix_t out = {0};
    matrix_init(&out, 1, 1);

    char img [60][30] = {0};

    for(size_t x = 0; x < 60; ++x) {
	for(size_t y = 0; y < 30; ++y) {

	    float inX = ((float)(x) - 20.0f) / 10.0f;
	    float inY = ((float)(y) - 10.0f) / 10.0f;
    	    matrix_set(&in, 0, 0, inX);
    	    matrix_set(&in, 1, 0, inY);
    	    matrix_set(&in, 2, 0, 1.0f);

	    network_inference(&testNet, &in, &out);

	    MATRIX_TYPE outNum;
	    matrix_get(&out, 0, 0, &outNum);

	    char outChar;
	    if(outNum < 0.04)
		outChar = '.';
	    else if(outNum < 0.08)
		outChar = ':';
	    else if(outNum < 0.12)
		outChar = '"';
	    else
		outChar = '#';
	    img[x][y] = outChar;
	}
    }

    /* Printing resulting map */
    for(size_t y = 0; y < 30; ++y) {
	for(size_t x = 0; x < 60; ++x) {
	    printf("%c ", img[x][y]);
	}
	putchar('\n');
    }

    /* Testing inference with tracking */
    MATRIX_TYPE n1 = 1.0f, n2 = 0.5f;
    matrix_set(&in, 0, 0, n1);
    matrix_set(&in, 1, 0, n2);
    matrix_set(&in, 2, 0, 1.0f);

    network_tracker_t testTracker = {0};
    network_tracker_init(&testTracker, testNet.depth, layers);

    network_inference_track(&testNet, &in, &out, &testTracker);

    puts("Network internal nodes:");
    puts("Layer 1:");
    printf("(%.2f, %.2f)\n(%.2f, %.2f)\n(%.2f, %.2f)\n(%.2f, %.2f)\n", 
	    testTracker.layerData[0][0][0], testTracker.layerData[0][0][1],
	    testTracker.layerData[0][1][0], testTracker.layerData[0][1][1],
	    testTracker.layerData[0][2][0], testTracker.layerData[0][2][1],
	    testTracker.layerData[0][3][0], testTracker.layerData[0][3][1]);
    puts("Layer 2:");
    printf("(%.2f, %.2f)\n", testTracker.layerData[1][0][0], testTracker.layerData[1][0][1]);

    network_tracker_destroy(&testTracker);

    //matrix_destroy(&in);
    //matrix_destroy(&out);

    network_destroy(&testNet);

    /* Testing training on dummy dataset */
    network_t trainTest = {0};
    size_t layers2 [2] = {3, 1};
    activation_t activations2 [2] = { activation_relu, activation_logistic };
    network_init(&trainTest, 3, 2, layers2, activations2);
    network_initWeights(&trainTest);

    puts("Randomized weights:");
    for(size_t idx = 0; idx < trainTest.depth; ++idx) {
	matrix_print(trainTest.weights + idx);
	putchar('\n');
    }

    set_t trainSet = {0};
    set_init(&trainSet, 10, 3, 1);
    MATRIX_TYPE inTrain [3] = {1.0f, 1.0f, 1.0f};
    MATRIX_TYPE outTrain [1] = {1.0f};
    set_setData(&trainSet, 0, inTrain, outTrain);
    inTrain[0] = 0.6f; inTrain[1] = 0.6f; inTrain[2] = 1.0f;
    outTrain[0] = 1.0f;
    set_setData(&trainSet, 1, inTrain, outTrain);
    inTrain[0] = 0.6f; inTrain[1] = 1.0f; inTrain[2] = 1.0f;
    outTrain[0] = 1.0f;
    set_setData(&trainSet, 2, inTrain, outTrain);
    inTrain[0] = 1.0f; inTrain[1] = 0.6f; inTrain[2] = 1.0f;
    outTrain[0] = 1.0f;
    set_setData(&trainSet, 3, inTrain, outTrain);
    inTrain[0] = 0.8f; inTrain[1] = 0.8f; inTrain[2] = 1.0f;
    outTrain[0] = 1.0f;
    set_setData(&trainSet, 4, inTrain, outTrain);

    inTrain[0] = 0.4f; inTrain[1] = 1.2f; inTrain[2] = 1.0f;
    outTrain[0] = 0.0f;
    set_setData(&trainSet, 5, inTrain, outTrain);
    inTrain[0] = 1.2f; inTrain[1] = 1.2f; inTrain[2] = 1.0f;
    outTrain[0] = 0.0f;
    set_setData(&trainSet, 6, inTrain, outTrain);
    inTrain[0] = 1.2f; inTrain[1] = 0.4f; inTrain[2] = 1.0f;
    outTrain[0] = 0.0f;
    set_setData(&trainSet, 7, inTrain, outTrain);
    inTrain[0] = 0.4f; inTrain[1] = 0.4f; inTrain[2] = 1.0f;
    outTrain[0] = 0.0f;
    set_setData(&trainSet, 8, inTrain, outTrain);
    inTrain[0] = 0.0f; inTrain[1] = 0.0f; inTrain[2] = 1.0f;
    outTrain[0] = 0.0f;
    set_setData(&trainSet, 9, inTrain, outTrain);

    set_train(&trainSet, &trainTest, layers2, 10, 10000);

    puts("Trained weights:");
    for(size_t idx = 0; idx < trainTest.depth; ++idx) {
	matrix_print(trainTest.weights + idx);
	putchar('\n');
    }

    /* Testing weights on heatmap */
    for(size_t x = 0; x < 60; ++x) {
	for(size_t y = 0; y < 30; ++y) {

	    float inX = ((float)(x) - 20.0f) / 10.0f;
	    float inY = ((float)(y) - 10.0f) / 10.0f;
    	    matrix_set(&in, 0, 0, inX);
    	    matrix_set(&in, 1, 0, inY);
    	    matrix_set(&in, 2, 0, 1.0f);

	    network_inference(&trainTest, &in, &out);

	    MATRIX_TYPE outNum;
	    matrix_get(&out, 0, 0, &outNum);

	    char outChar;
	    if(outNum < 0.04)
		outChar = '.';
	    else if(outNum < 0.08)
		outChar = ':';
	    else if(outNum < 0.12)
		outChar = '"';
	    else
		outChar = '#';
	    img[x][y] = outChar;
	}
    }

    /* Printing resulting map */
    for(size_t y = 0; y < 30; ++y) {
	for(size_t x = 0; x < 60; ++x) {
	    printf("%c ", img[x][y]);
	}
	putchar('\n');
    }


    set_destroy(&trainSet);
    network_destroy(&trainTest);

    matrix_destroy(&in);
    matrix_destroy(&out);

    return 0;
}
