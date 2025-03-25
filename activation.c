#include "activation.h"

void activation_relu(matrix_t * m) {
    for(size_t idx = 0; idx < m->dataLen; ++idx) {
	MATRIX_TYPE res = m->data[idx];
	m->data[idx] = (res > 0 ? res : RELU_LEAK * res);
    }
}

void activation_logistic(matrix_t * m) {
    for(size_t idx = 0; idx < m->dataLen; ++idx) {
	m->data[idx] = 1 / (1 + exp(-1 * m->data[idx]));
    }
}
