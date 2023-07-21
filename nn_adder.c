#include <time.h>
#define NN_IMPLEMENTATION
#include "lib/nn.h"

// Till BITS < 4   This works               =>  size_t arch[] = {2*BITS, 2*BITS, BITS + 1}
// Till BITS = 4   This needs to be used    =>  size_t arch[] = {2*BITS, 4*BITS, BITS + 1}
#define BITS 4
//#define FULL_DISPLAY 1

int main()
{
    // srand(time(0));
    srand(69);
    
    size_t n = (1<<BITS);
    printf("BITS : %d, Max Value : %zu\n", BITS, n);
    size_t rows = n*n;
    
    Mat ti = mat_alloc(rows, 2*BITS);
    Mat to = mat_alloc(rows, BITS + 1);
    for(size_t i = 0; i < ti.rows; i++)
    {
        size_t x = i/n;
        size_t y = i%n;
        size_t z = x + y;
        for(size_t j = 0; j < BITS; j++)
        {
            MAT_AT(ti, i, j)        = (x>>j)&1;
            MAT_AT(ti, i, j + BITS) = (y>>j)&1;
            MAT_AT(to, i, j)        = (z>>j)&1;
        }
        MAT_AT(to, i, BITS) = z >= n;
    }
    
    size_t arch[] = {2*BITS, 2*BITS, BITS + 1};
    NN nn = nn_alloc(arch, ARRAY_LEN(arch));
    NN g  = nn_alloc(arch, ARRAY_LEN(arch));
    nn_rand(nn, 0, 1);
    
    NN_PRINT(nn);
    float rate = 1;
    
    printf("Cost : %f\n", nn_cost(nn, ti, to));
    for(size_t i = 0; i < 10*1000; i++)
    {
        nn_backprop(nn, g, ti, to);
        nn_learn(nn, g, rate);
        printf("%zu : Cost : %f\n", i, nn_cost(nn, ti, to));
    }
    
    printf("===========================================\n");
    size_t fails = 0;
    for(size_t x = 0; x < n; x++)
    {
        for(size_t y = 0; y < n; y++)
        {
            #ifdef FULL_DISPLAY
            printf("%zu + %zu = ", x, y);
            #endif
            for(size_t j = 0; j < BITS; j++)
            {
                MAT_AT(NN_INPUT(nn), 0, j)        = (x>>j)&1;
                MAT_AT(NN_INPUT(nn), 0, j + BITS) = (y>>j)&1;
            }
            nn_forward(nn);
            size_t z = 0;
            for(size_t j = 0; j < BITS; j++)
            {
                size_t bit = MAT_AT(NN_OUTPUT(nn), 0, j) > 0.5f;
                z |= bit<<j;
            }
            size_t overflow = MAT_AT(NN_OUTPUT(nn), 0, BITS) > 0.5f;

            size_t a = x + y;
            if(overflow)
            {
                #ifdef FULL_DISPLAY               
                if(a < n)
                {
                    printf("%zu [ERROR]\n", z);
                    fails += 1;
                }
                else
                {
                    printf("%zu [OVERFLOW]\n", z);
                }
                #else
                if(a < n)
                {
                    printf("%zu + %zu = [OVERFLOW <> %zu]\n", x, y, a);
                    fails += 1;
                }
                #endif
            }
            else
            {
                #ifdef FULL_DISPLAY
                if(a != z)
                {
                    printf("%zu [ERROR %zu <> %zu]\n", z, a, z);
                    fails += 1;
                }
                else
                {
                    printf("%zu\n", z);
                }
                #else
                if(a != z)
                {
                    printf("%zu + %zu = [%zu <> %zu]\n", x, y, z, a);
                    fails += 1;
                }
                #endif
            }
        }
    }
    
    if(fails == 0)
    {
        printf("===========================================\nAll Ok.");
    }
    else
    {
        printf("===========================================\n%zu Error(s).", fails);
    }
    
    return 0;
}
