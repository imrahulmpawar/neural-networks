#!/bin/sh
set -xe
clear
clang -Wall -Wextra -o /data/data/com.termux/files/home/main storage/shared/Documents/neuron/nn_framework_xor.c -lm