#include <cstdlib>
#include "../include/add.h"

extern "C" int main_driver(int argc, char **argv) {
    addHandle_t handle;
    int err;
    int I = 42;
    int J = 42;
    int K = 42;
    float * __restrict__ __return = (float*) calloc(((I * J) * K), sizeof(float));
    float * __restrict__ a = (float*) calloc(((I * J) * K), sizeof(float));
    float * __restrict__ b = (float*) calloc(((I * J) * K), sizeof(float));


    handle = __dace_init_add(I, J, K);
    __program_add(handle, __return, a, b, I, J, K);
    err = __dace_exit_add(handle);

    free(__return);
    free(a);
    free(b);


    return err;
}