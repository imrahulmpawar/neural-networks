#!/bin/sh
set -xe
clear
clang -Wall -Wextra -o /data/data/com.termux/files/home/main storage/shared/Documents/neuron/gates_gradient.c -lm