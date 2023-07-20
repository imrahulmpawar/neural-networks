#!/bin/sh
set -xe
export PATH=/usr/local/clang_9.0.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/clang_9.0.0/lib:$LD_LIBRARY_PATH
clang -Wall -Wextra -o $1.out $1.c -lm