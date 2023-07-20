#include <stdio.h>
#include <stdlib.h>
#include <time.h>

float train[][2] = {
    {0, 1},
    {1, 2},
    {2, 4},
    {3, 6},
    {4, 8},
};
#define TRAIN_COUNT (sizeof(train)/sizeof(train[0]))

float rand_float()
{
    return (float)rand()/(float)RAND_MAX;
}

float cost(float w)
{
    float result = 0.0f;
    for(size_t i=0; i<TRAIN_COUNT; i++)
    {
        float x = train[i][0];
        float y = x*w;
        float e = y - train[i][1];
        result += e*e;
        // printf("Actual: %f, Expected: %f\n", y, train[i][1]);
    }
    result /= TRAIN_COUNT;
    
    return result;
}

float dcost(float w)
{
    // derivative of cost function
    // d(cost) = (2(x.w -y).x)/n
    
    float result = 0.0f;
    for(size_t i=0; i<TRAIN_COUNT; i++)
    {
        float x = train[i][0];
        float y = train[i][1];
        result += 2*(x*w - y)*x;
    }
    result /= TRAIN_COUNT;
    
    return result;
}

int main()
{
    // y = x*w
    //srand(time(0));
    srand(69);
    
    float w1 = rand_float()*10.0f;
    float w2 = w1;
    float eps = 1e-1;
    float rate = 1e-1;
    int iterations = 10;
    
    printf(" cost: %f, w1: %f | gcost: %f, w2: %f\n", cost(w1), w1, cost(w2), w2);
    printf("-----------------------\n");
    float dw1, dw2, c = 0.0f;
    for(int i=0; i<iterations; i++)
    {
        c = cost(w1);
        dw1 = (cost(w1 + eps) - c)/eps;
        w1 -= rate*dw1;
        
        dw2 = dcost(w2);
        w2 -= rate*dw2;
        printf("%d: cost: %f, w1: %f | gcost: %f, w2: %f\n", i, cost(w1), w1, cost(w2), w2);
    }
    
    printf("-----------------------\n");
    printf(" cost: %f, w1: %f | gcost: %f, w2: %f\n", cost(w1), w1, cost(w2), w2);
    
    return 0;
}