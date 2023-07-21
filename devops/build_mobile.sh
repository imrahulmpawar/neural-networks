#!/bin/sh
set -xe
clear
clang -Wall -Wextra -o ./main.out ./storage/shared/Documents/neuron/gates_gradient.c -lm
