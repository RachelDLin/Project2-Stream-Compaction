#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

/*****************
* Configuration *
*****************/

/*! Block size used for CUDA kernel launch. */
#define blockSize 256

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        
        __global__ void naiveParallelScanKernel(int n, int* odata, const int* idata, int k) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;

            if (index < n) {
                int kernelStart = 1 << (k - 1);

                if (index >= kernelStart) {
                    odata[index] = idata[index] + idata[index - kernelStart];
                } else {
                    odata[index] = idata[index];
                }
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // use global memory
            // ilog2ceil(n) kernel invocations
            // since GPU threads are not guaranteed to run simultaneously, we can't operate on an array in-place on the GPU (it will cause race conditions)
            // instead, create two device arrays and swap them every iteration

            // device array buffers
            int* dev_arrA;
            int* dev_arrB;

            // allocate buffers
            cudaMalloc((void**)&dev_arrA, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_arrA failed!");
            cudaMalloc((void**)&dev_arrB, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_arrB failed!");

            cudaMemcpy(dev_arrA, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
            
            timer().startGpuTimer();
            
            // for each kernel
            for (int k = 1; k <= ilog2ceil(n); k++) {
                // call naive scan kernel
                naiveParallelScanKernel << <fullBlocksPerGrid, blockSize >> > (n, dev_arrB, dev_arrA, k);
                checkCUDAError("naiveParallelScanKernel failed!");

                // swap array buffers
                std::swap(dev_arrA, dev_arrB);
            }
            cudaDeviceSynchronize();

            timer().endGpuTimer();

            // copy data back to the CPU side from the GPU
            odata[0] = 0;
            cudaMemcpy(odata + 1, dev_arrA, (n - 1) * sizeof(int), cudaMemcpyDeviceToHost);

            // cleanup
            cudaFree(dev_arrA);
            cudaFree(dev_arrB);
        }
    }
}
