#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

// OR Gate
float train[][3] = {
    {0, 0, 0},
    {1, 0, 1},
    {0, 1, 1},
    {1, 1, 1},
};
#define TRAIN_COUNT (sizeof(train)/sizeof(train[0]))

float sigmoidf(float x)
{
    return 1.0f/(1 + expf(-x));
}

float rand_float()
{
    return (float)rand()/(float)RAND_MAX;
}

float cost(float w1, float w2, float b)
{
    float result = 0.0f;
    for(size_t i=0; i<TRAIN_COUNT; i++)
    {
        float x1 = train[i][0];
        float x2 = train[i][1];
        float y = sigmoidf(x1*w1 + x2*w2 + b);
        float e = y - train[i][2];
        result += e*e;
        // printf("Actual: %f, Expected: %f\n", y, train[i][1]);
    }
    result /= TRAIN_COUNT;
    
    return result;
}

int main()
{
    // y = x*w
    // srand(time(0));
    srand(69);
    
    float w1 = rand_float();
    float w2 = rand_float();
    float b = rand_float();
    
    float eps = 1e-1;
    float rate = 1e-1;
    int iterations = 100000;
    
    printf(" cost: %f, w1: %f, w2: %f, b: %f\n", cost(w1, w2, b), w1, w2, b);
    float dw1, dw2, db, c = 0.0f;
    for(int i=0; i<iterations; i++)
    {
        c = cost(w1, w2, b);
        dw1 = (cost(w1 + eps, w2, b) - c)/eps;
        dw2 = (cost(w1, w2 + eps, b) - c)/eps;
        db = (cost(w1, w2, b + eps) - c)/eps;
        
        w1 -= rate*dw1;
        w2 -= rate*dw2;
        b -= rate*db;
        printf(" cost: %f, w1: %f, w2: %f, b: %f\n", cost(w1, w2, b), w1, w2, b);
    }
   
    printf("-----------------------\n");
    for(int i=0; i<TRAIN_COUNT; i++)
    {
        int x1 = train[i][0];
        int x2 = train[i][1];
        printf(" %d x %d = %f\n", x1, x2, sigmoidf(x1*w1 + x2*w2 + b));
    }
    
    return 0;
}