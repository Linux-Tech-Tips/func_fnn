#!/bin/bash
gcc -lc -lm -std=gnu99 -Wall -Werror -pedantic -o matrix_test.elf main.c matrix.c activation.c network.c set.c util.c
