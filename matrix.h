/** 
 * @file matrix.h
 * @author Linux-Tech-Tips (Martin)
 * @brief File containing Matrix data structure and functions
 */
#ifndef MATRIX_H
#define MATRIX_H

#include <stdlib.h>
#include <stdio.h>

#ifndef MATRIX_TYPE
#define MATRIX_TYPE float
#endif

#ifndef MATRIX_TYPE_F
#define MATRIX_TYPE_F "%.2f"
#endif

/** Structure containing data for a matrix */
typedef struct {

    MATRIX_TYPE * data;
    size_t dataLen;

    size_t rows;
    size_t cols;

} matrix_t;

/** Matrix file error types, returned from matrix functions */
typedef enum {
    /** Default state, op successful */
    MATRIX_OK = 0,
    /** Invalid index error */
    MATRIX_ERR_INDEX = 1,
    /** Error allocating data for the Matrix */
    MATRIX_ERR_ALLOC = 2,
    /** Error on binary operation - two given matrices incompatible */
    MATRIX_ERR_MISMATCH = 3,
    /** General/unspecified matrix error */
    MATRIX_ERR = 4
} matrix_err_t;


/** Calculate the flat index of a matrix based on its parameters */
matrix_err_t _matrix_flatIdx(size_t row, size_t col, size_t rows, size_t cols, size_t * idx);

/** Initialise a matrix structure of the given size (must be destroyed after) */
matrix_err_t matrix_init(matrix_t * m, size_t rows, size_t cols);

/** Destroys an initialised matrix which will no longer be used */
matrix_err_t matrix_destroy(matrix_t * m);

/** Return an element at the given row and column of a matrix */
matrix_err_t matrix_get(matrix_t * m, size_t row, size_t col, MATRIX_TYPE * val);

/** Set an element at the given row and column of a matrix */
matrix_err_t matrix_set(matrix_t * m, size_t row, size_t col, MATRIX_TYPE val);

/** Copy content of m1 into m2 */
matrix_err_t matrix_copy(matrix_t * m1, matrix_t * m2);

/** Multiply two matrices, save the result into the third (result = m1*m2) - all three matrices must have appropriate dimensions */
matrix_err_t matrix_matmul(matrix_t * m1, matrix_t * m2, matrix_t * result);

/** Prints the matrix in a readable way */
matrix_err_t matrix_print(matrix_t * m);

#endif /* MATRIX_H */
