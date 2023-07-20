#include <time.h>
#define NN_IMPLEMENTATION
#include "nn.h"

int main()
{
    srand(time(0));

    Mat a = mat_alloc(1, 2);
    mat_rand(a, 5, 10);

    Mat b = mat_alloc(2, 2);
    mat_identity(b);

    Mat dst = mat_alloc(1, 2);
    
    mat_print(a);
    printf("---------------------\n");
    mat_print(b);
    printf("---------------------\n");
    mat_dot(dst, a, b);
    mat_print(dst);
    
    return 0;
}


