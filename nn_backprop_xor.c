#include <time.h>
#define NN_IMPLEMENTATION
#include "lib/nn.h"

float td_xor[] = {
    0, 0, 0,
    0, 1, 1,
    1, 0, 1,
    1, 1, 0,
};

float td_or[] = {
    0, 0, 0,
    0, 1, 1,
    1, 0, 1,
    1, 1, 1,
};

int main()
{
    // srand(time(0));
    srand(69);
    
    float *td = td_or;

    size_t stride = 3;
    size_t n = 4; //sizeof(td)/sizeof(td[0])/stride;
    
    Mat ti = {
        .rows = n,
        .cols = 2,
        .stride = stride,
        .es = td,
    };
    
    Mat to = {
        .rows = n,
        .cols = 1,
        .stride = stride,
        .es = td + 2,
    };
    
    size_t arch[] = {2, 2, 1};
    NN nn = nn_alloc(arch, ARRAY_LEN(arch));
    NN g = nn_alloc(arch, ARRAY_LEN(arch));
    
    float rate = 1;

    nn_rand(nn, 0, 1);
    
    printf("cost = %f\n", nn_cost(nn, ti, to));
    for(size_t i = 0; i < 5000; i++)
    {
        nn_backprop(nn, g, ti, to);
        nn_learn(nn, g, rate);
        printf("cost = %f\n", nn_cost(nn, ti, to));
    }
					
    printf("-------------------------\n");

    NN_PRINT(nn);

    for(size_t i = 0; i < 2; i++)
    {
        for(size_t j = 0; j < 2; j++)
        {
            MAT_AT(NN_INPUT(nn), 0, 0) = i;
            MAT_AT(NN_INPUT(nn), 0, 1) = j;
            nn_forward(nn);
            float y = MAT_AT(NN_OUTPUT(nn), 0, 0);
            
            printf("%zu ^ %zu = %f\n", i, j, y);
        }
    }
    
    return 0;
}


