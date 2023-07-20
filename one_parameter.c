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

float cost(float w, float b)
{
    float result = 0.0f;
    for(size_t i=0; i<TRAIN_COUNT; i++)
    {
        float x = train[i][0];
        float y = x*w + b;
        float e = y - train[i][1];
        result += e*e;
        // printf("Actual: %f, Expected: %f\n", y, train[i][1]);
    }
    result /= TRAIN_COUNT;
    
    return result;
}

int main()
{
    // y = x*w
    srand(time(0));
    // srand(69);
    
    float w = rand_float()*10.0f;
    
    float b = rand_float()*5.0f;
    float eps = 1e-3;
    float rate = 1e-3;
    int iterations = 500;
    
    printf(" cost: %f, w: %f, b: %f\n", cost(w, b), w, b);
    float dw, db, c = 0.0f;
    for(int i=0; i<iterations; i++)
    {
        c = cost(w, b);
        dw = (cost(w + eps, b) - c)/eps;
        db = (cost(w, b + eps) - c)/eps;
        w -= rate*dw;
        b -= rate*db;
        printf(" cost: %f, w: %f, b: %f\n", cost(w, b), w, b);
    }
    
    printf("-----------------------\n");
    printf(" cost: %f, w: %f, b: %f\n", cost(w, b), w, b);
    
    return 0;
}