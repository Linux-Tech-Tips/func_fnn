#include "activation.h"

void activation_relu_f(matrix_t * m) {
    for(size_t idx = 0; idx < m->dataLen; ++idx) {
	MATRIX_TYPE res = m->data[idx];
	m->data[idx] = (res > 0 ? res : RELU_LEAK * res);
    }
}

void activation_relu_df(matrix_t * m) {
    for(size_t idx = 0; idx < m->dataLen; ++idx) {
	MATRIX_TYPE res = m->data[idx];
	m->data[idx] = (res > 0 ? 1 : RELU_LEAK);
    }
}

activation_t activation_relu = {
    .f = activation_relu_f,
    .df = activation_relu_df
};

void activation_logistic_f(matrix_t * m) {
    for(size_t idx = 0; idx < m->dataLen; ++idx) {
	m->data[idx] = 1.0f / (1 + exp(-1 * m->data[idx]));
    }
}

void activation_logistic_df(matrix_t * m) {
    for(size_t idx = 0; idx < m->dataLen; ++idx) {
	MATRIX_TYPE res = m->data[idx];
	m->data[idx] = 1.0f / (exp(res) * (1 + exp(-1 * res) * (1 + exp(-1 * res))));
    }
}

activation_t activation_logistic = {
    .f = activation_logistic_f,
    .df = activation_logistic_df
};
