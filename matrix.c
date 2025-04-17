#include "matrix.h"

matrix_err_t _matrix_flatIdx(size_t row, size_t col, size_t rows, size_t cols, size_t * idx) {
    matrix_err_t result = MATRIX_ERR_INDEX;
    if(row < rows && col < cols) {
	*idx = (row * cols) + col;
	result = MATRIX_OK;
    }
    return result;
}

matrix_err_t matrix_init(matrix_t * m, size_t rows, size_t cols) {
    matrix_err_t result = MATRIX_ERR_ALLOC;
    if(m->data == NULL && m->dataLen == 0) {
	m->dataLen = rows * cols;
	m->data = (MATRIX_TYPE *)(malloc(m->dataLen * sizeof(MATRIX_TYPE)));
	m->rows = rows;
	m->cols = cols;
	result = MATRIX_OK;
    }
    return result;
}

matrix_err_t matrix_destroy(matrix_t * m) {
    matrix_err_t result = MATRIX_ERR_ALLOC;
    if(m->data != NULL) {
        free(m->data);
	m->data = NULL;
	m->dataLen = 0;
	result = MATRIX_OK;
    }
    return result;
}


matrix_err_t matrix_get(matrix_t * m, size_t row, size_t col, MATRIX_TYPE * val) {
    size_t idx = 0;
    matrix_err_t result = _matrix_flatIdx(row, col, m->rows, m->cols, &idx);
    if(result == MATRIX_OK) {
	*val = m->data[idx];
    }
    return result;
}

matrix_err_t matrix_set(matrix_t * m, size_t row, size_t col, MATRIX_TYPE val) {
    size_t idx = 0;
    matrix_err_t result = _matrix_flatIdx(row, col, m->rows, m->cols, &idx);
    if(result == MATRIX_OK) {
	m->data[idx] = val;
    }
    return result;
}

matrix_err_t matrix_populate(matrix_t * m, MATRIX_TYPE (*val)(size_t idx)) {
    if(!m || !val)
	return MATRIX_ERR;
    for(size_t i = 0; i < m->dataLen; ++i) {
	m->data[i] = val(i);
    }
    return MATRIX_OK;
}

matrix_err_t matrix_copy(matrix_t * m1, matrix_t * m2) {
    matrix_err_t result = MATRIX_ERR_MISMATCH;
    if(m1->cols == m2->cols && m1->rows == m2->rows && m1->dataLen == m2->dataLen) {
	for(size_t idx = 0; idx < m1->dataLen; ++idx) {
	    m2->data[idx] = m1->data[idx];
	}
	result = MATRIX_OK;
    }
    return result;
}

matrix_err_t matrix_matmul(matrix_t * m1, matrix_t * m2, matrix_t * result) {
    /* Check appropriate dimensions */
    if(!(m1->cols == m2->rows && result->rows == m1->rows && result->cols == m2->cols)) {
	return MATRIX_ERR_MISMATCH;
    }
    /* Do the multiplication */
    for(size_t row = 0; row < m1->rows; ++row) {
	for(size_t col = 0; col < m2->cols; ++col) {
	    /* Dot product of corresponding row and column of m1 and m2 */
	    MATRIX_TYPE resVal = 0;
	    /* Here, m1->cols is guaranteed to be equal to m2->rows, so we can index both the m1 column and m2 row */
	    for(size_t idx = 0; idx < m1->cols; ++idx) {
		MATRIX_TYPE val1;
		matrix_get(m1, row, idx, &val1);
		MATRIX_TYPE val2;
		matrix_get(m2, idx, col, &val2);
		resVal += val1 * val2;
	    }
	    matrix_set(result, row, col, resVal);
	}
    }
    return MATRIX_OK;
}

matrix_err_t matrix_print(matrix_t * m) {
    for(size_t row = 0; row < m->rows; ++row) {
	for(size_t col = 0; col < m->cols; ++col) {
	    MATRIX_TYPE val;
	    matrix_get(m, row, col, &val);
	    printf(MATRIX_TYPE_PRINTF "\t", val);
	}
	putchar('\n');
    }
    return MATRIX_OK;
}
