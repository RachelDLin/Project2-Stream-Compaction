#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

/*****************
* Configuration *
*****************/

/*! Block size used for CUDA kernel launch. */
#define blockSize 256


namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }



        // up-sweep
        __global__ void kernUpsweep(int n, int d, int* idata) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            int stride = 1 << (d + 1); // 2^(d+1)

            if (index < n) {
                int idx1 = index * stride + stride - 1;;
                int idx2 = idx1 - (stride >> 1);
                idata[idx1] += idata[idx2];
            }
        }

        // down-sweep
        __global__ void kernDownsweep(int n, int d, int* idata) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            int stride = 1 << (d + 1); // 2^(d+1)

            if (index < n) {
                int idx1 = index * stride + stride - 1;
                int idx2 = idx1 - (stride >> 1);

                int t = idata[idx2];
                idata[idx2] = idata[idx1];
                idata[idx1] += t;
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            
            // set new n
            int newN = 1 << ilog2ceil(n);
            
            // device array buffer
            int* dev_arr;

            // allocate buffers
            cudaMalloc((void**)&dev_arr, newN * sizeof(int));
            checkCUDAError("cudaMalloc dev_arr failed!");

            cudaMemset(dev_arr, 0, newN * sizeof(int));
            cudaMemcpy(dev_arr, idata, n * sizeof(int), cudaMemcpyHostToDevice);



            timer().startGpuTimer();

            // upsweep
            for (int d = 0; d < ilog2ceil(n); d++) {

                int numNodes = newN >> (d + 1);

                dim3 fullBlocksPerGrid((numNodes + blockSize - 1) / blockSize);

                kernUpsweep << <fullBlocksPerGrid, blockSize >> > (newN, d, dev_arr);

                checkCUDAError("kernUpsweep failed!");
                cudaDeviceSynchronize();
            }

            // downsweep
            cudaMemset(dev_arr + (newN - 1), 0, sizeof(int));
            for (int d = ilog2ceil(n) - 1; d >= 0; d--) {

                int numNodes = newN >> (d + 1);

                dim3 fullBlocksPerGrid((numNodes + blockSize - 1) / blockSize);

                kernDownsweep << <fullBlocksPerGrid, blockSize >> > (newN, d, dev_arr);

                checkCUDAError("kernDownsweep failed!");
                cudaDeviceSynchronize();
            }
            cudaDeviceSynchronize();

            timer().endGpuTimer();

            // copy data back to the CPU side from the GPU
            cudaMemcpy(odata, dev_arr, n * sizeof(int), cudaMemcpyDeviceToHost);

            // cleanup
            cudaFree(dev_arr);
        }


        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata  (device only).
         */
        void compactScan(int n, int* dev_odata, const int* dev_idata) {

            // set new n
            int newN = 1 << ilog2ceil(n);

            // device array buffer
            int* dev_arr;
            
            // allocate buffers
            cudaMalloc((void**)&dev_arr, newN * sizeof(int));
            checkCUDAError("cudaMalloc dev_arr failed!");

            cudaMemset(dev_arr, 0, newN * sizeof(int));
            cudaMemcpy(dev_arr, dev_idata, n * sizeof(int), cudaMemcpyDeviceToDevice);


            // upsweep
            for (int d = 0; d < ilog2ceil(n); d++) {
                int numNodes = newN / (1 << (d + 1));

                dim3 fullBlocksPerGrid((numNodes + blockSize - 1) / blockSize);

                kernUpsweep << <fullBlocksPerGrid, blockSize >> > (numNodes, d, dev_arr);

                checkCUDAError("kernUpsweep failed!");
                cudaDeviceSynchronize();
            }

            cudaMemset(dev_arr + (newN - 1), 0, sizeof(int));

            // downsweep
            for (int d = ilog2ceil(n) - 1; d >= 0; d--) {
                int numNodes = newN / (1 << (d + 1));

                dim3 fullBlocksPerGrid((numNodes + blockSize - 1) / blockSize);

                kernDownsweep << <fullBlocksPerGrid, blockSize >> > (numNodes, d, dev_arr);

                checkCUDAError("kernDownsweep failed!");
                cudaDeviceSynchronize();
            }
            cudaDeviceSynchronize();

            // copy data back to the CPU side from the GPU
            cudaMemcpy(dev_odata, dev_arr, n * sizeof(int), cudaMemcpyDeviceToDevice);

            // cleanup
            cudaFree(dev_arr);
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            int newN = 1 << ilog2ceil(n);

            // allocate buffers
            int* dev_idata;
            int* dev_odata;
            int* dev_isValid;
            int* dev_indices;

            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed!");
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_odata failed!");
            cudaMalloc((void**)&dev_isValid, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_isValid failed!");
            cudaMalloc((void**)&dev_indices, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_indices failed!");

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);


            timer().startGpuTimer();

            // temporary array to indicate the element should be kept/discarded
            StreamCompaction::Common::kernMapToBoolean << <fullBlocksPerGrid, blockSize >> > (n, dev_isValid, dev_idata);

            // exclusive scan on temporary array
            compactScan(n, dev_indices, dev_isValid);
            
            // scatter
            StreamCompaction::Common::kernScatter << <fullBlocksPerGrid, blockSize >> > (n, dev_odata, dev_idata, dev_isValid, dev_indices);
            cudaDeviceSynchronize();

            timer().endGpuTimer();


            // copy data from dev_odata to odata
            int lastValid = 0;
            cudaMemcpy(&lastValid, dev_isValid + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            int lastIdx = 0;
            cudaMemcpy(&lastIdx, dev_indices + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            int numElems = lastValid + lastIdx;
            cudaMemcpy(odata, dev_odata, numElems * sizeof(int), cudaMemcpyDeviceToHost);

            // cleanup
            cudaFree(dev_idata);
            cudaFree(dev_odata);
            cudaFree(dev_indices);
            cudaFree(dev_isValid);

            return numElems;
        }
    }
}
