#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include "common.h"
#include "thrust.h"

namespace StreamCompaction {
    namespace Thrust {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {

            // use `thrust::exclusive_scan`
            // example: for device_vectors dv_in and dv_out:
            // thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());

            if (n > 0) {
                // copy input from host pointer to host vector 
                //thrust::host_vector<int> host_iData(idata, idata + n);

                // cast as device vector
                thrust::device_vector<int> dev_iData(idata, idata + n);

                // device output vector
                thrust::device_vector<int> dev_oData(n);

                timer().startGpuTimer();

                // perform exclusive scan on GPU
                thrust::exclusive_scan(dev_iData.begin(), dev_iData.end(), dev_oData.begin());

                timer().endGpuTimer();

                // copy result to host output vector
                thrust::copy(dev_oData.begin(), dev_oData.end(), odata);
            }
        }
    }
}
