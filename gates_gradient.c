#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

// OR Gate
typedef float sample[3];

sample or_train[] = {
    {0, 0, 0},
    {1, 0, 1},
    {0, 1, 1},
    {1, 1, 1},
};

sample and_train[] = {
    {0, 0, 0},
    {1, 0, 0},
    {0, 1, 0},
    {1, 1, 1},
};

sample nand_train[] = {
    {0, 0, 1},
    {1, 0, 0},
    {0, 1, 0},
    {1, 1, 0},
};

#define TRAIN_COUNT 4
sample *train = and_train;

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

void gcost(float w1, float w2, float b, float *dw1, float *dw2, float *db)
{
    *dw1 = 0;
    *dw2 = 0;
    *db  = 0;
    
    for(size_t i=0; i<TRAIN_COUNT; i++)
    {
        float x1 = train[i][0];
        float x2 = train[i][1];
        float y = train[i][2];
        
        float a = sigmoidf(x1*w1 + x2*w2 + b);
        float a_com = 2*(a - y)*a*(1 - a);
        *dw1 += a_com*x1;
        *dw2 += a_com*x2;
        *db += a_com;
    }
    *dw1 /= TRAIN_COUNT;
    *dw2 /= TRAIN_COUNT;
    *db /= TRAIN_COUNT;
}

#define FINITE_DIFF 1
int main()
{
    // y = x*w
    // srand(time(0));
    srand(69);
    
    float w1 = rand_float();
    float w2 = rand_float();
    float b = rand_float();
    
    float rate = 1e-1;
#ifdef FINITE_DIFF
    int iterations = 100000;
    float eps = 1e-1;
    float c = 0.f;
#else
    int iterations = 1000;
#endif
    iterations = 1000;

    printf(" cost: %f, w1: %f, w2: %f, b: %f\n", cost(w1, w2, b), w1, w2, b);
    float dw1, dw2, db = 0.0f;
    for(int i=0; i<iterations; i++)
    {
#ifdef FINITE_DIFF
        c = cost(w1, w2, b);
        dw1 = (cost(w1 + eps, w2, b) - c)/eps;
        dw2 = (cost(w1, w2 + eps, b) - c)/eps;
        db = (cost(w1, w2, b + eps) - c)/eps;
#else
        gcost(w1, w2, b, &dw1, &dw2, &db);
#endif       
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