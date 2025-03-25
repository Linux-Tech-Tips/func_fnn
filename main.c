/* Main file, testing for now */

#include <stdio.h>

#include "matrix.h"
#include "activation.h"

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

    matrix_destroy(&m1);
    matrix_destroy(&m2);

    return 0;
}
