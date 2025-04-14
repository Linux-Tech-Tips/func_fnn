/* Main file, testing for now */

#include <stdio.h>

#include "matrix.h"
#include "activation.h"
#include "network.h"

int main(void) {

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

    a1(&result);
    matrix_print(&result);

    a2(&result);
    matrix_print(&result);

    activation_softmax(&result);
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

    matrix_destroy(&in);
    matrix_destroy(&out);

    network_destroy(&testNet);

    /* Printing resulting map */
    for(size_t y = 0; y < 30; ++y) {
	for(size_t x = 0; x < 60; ++x) {
	    printf("%c ", img[x][y]);
	}
	putchar('\n');
    }

    return 0;
}
