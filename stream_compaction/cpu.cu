#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();

            // check size of idata
            if (n == 0) {
                return;
            }
            
            // compute exclusive prefix sum (ignore last element)
            odata[0] = 0;

            for (int i = 1; i < n; i++) {
                odata[i] = odata[i - 1] + idata[i - 1];
            }

            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            
            // number of elements remaining
            int numElems = 0;
            
            // compaction
            for (int i = 0; i < n; i++) {
                if (idata[i] != 0) {
                    odata[numElems] = idata[i];
                    numElems++;
                }
            }

            timer().endCpuTimer();
            return numElems;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();

            // temporary array to indicate the element should be kept/discarded
            int* isValid = new int[n];
            
            for (int i = 0; i < n; i++) {
                isValid[i] = 0;
                if (idata[i] != 0) {
                    isValid[i] = 1;
                } 
            }

            // exclusive prefix sum scan on temp array
            // represents the index in odata that i in idata should be mapped to
            int* indices = new int[n];
            
            // compute exclusive prefix sum (ignore last element)
            indices[0] = 0;

            for (int i = 1; i < n; i++) {
                indices[i] = indices[i - 1] + isValid[i - 1];
            }

            // number of elements remaining
            int numElems = 0;

            // scatter
            for (int i = 0; i < n; i++) {
                if (isValid[i] == 1) {
                    int idx = indices[i];
                    odata[idx] = idata[i];
                    numElems++;
                }
            }

            timer().endCpuTimer();

            delete[] isValid;
            delete[] indices;

            return numElems;
        }
    }
}
