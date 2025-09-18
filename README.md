CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Rachel Lin

  * [LinkedIn](https://www.linkedin.com/in/rachel-lin-452834213/)
  * [personal website](https://www.artstation.com/rachellin4)
  * [Instagram](https://www.instagram.com/lotus_crescent/)

* Tested on: (TODO) Windows 11, 12th Gen Intel(R) Core(TM) i7-12700H @ 2.30GHz, NVIDIA GeForce RTX 3080 Laptop GPU (16 GB)


## Description

This project offers parallel scan and stream compaction algorithms in CUDA. Features include:
  * stream compaction to remove unwanted elements (zeros) from an input data array and scatter the valid elements into a compacted output buffer
  * exclusive (prefix sum) scanning
   * on the CPU using a simple for-loop
   * on the GPU using a naive algorithm
   * on the GPU using a work-efficient algorithm that avoids race conditions
   * on the GPU using thrust library

## Performance Analysis

### Comparison of GPU Scan Implementations

<img src="img/Scan Time vs. Array Size (Power of Two).png" width="50%">
<img src="img/Scan Time vs. Array Size (Non-Power of Two).png" width="50%">

#### Average Scan Time vs. Array Size (Power of Two)

| Array Size	| CPU	| Naive	| Work-Efficient	| Thrust | 
| --------- | --------- | --------- | --------- | --------- |
| 256	| 0.0006	| 0.1969493333	| 0.2730666667	| 0.1140053333 | 
| 1024	| 0.0018	| 0.2085546667	| 0.41984	| 0.1235733333 | 
| 16384	| 0.0284	| 0.254976	| 0.9103373333	| 0.1314133333 | 
| 131072	| 0.2284333333	| 0.6356693333	| 0.7360333333	| 0.130048 | 
| 1048576	| 1.737733333	| 1.173386667	| 1.053409333	| 0.759808 | 
| 4194304	| 7.669166667	| 4.442293333	| 2.438283333	| 0.8376226667 | 
| 16777216	| 27.9746	| 13.6956	| 6.76686	| 1.431213333 | 

#### Average Scan Time vs. Array Size (Non-Power of Two)

| Array Size	| CPU	| Naive	| Work-Efficient	| Thrust | 
| --------- | --------- | --------- | --------- | --------- |
| 253	| 0.0005	| 0.06144	| 0.2095786667	| 0.05768533333 | 
| 1021	| 0.0019	| 0.1505493333	| 0.2740906667	| 0.05290666667 | 
| 16383	| 0.03276666667	| 0.1723733333	| 0.372736	| 0.05563733333 | 
| 131069	| 0.2283666667	| 0.6361066667	| 0.5046293333	| 0.045056 | 
| 1048573	| 2.531466667	| 1.062026667	| 0.9203946667	| 0.75264 | 
| 4194301	| 7.606633333	| 5.455966667	| 3.874946667	| 1.58338 | 
| 16777213	| 28.6349	| 13.5973	| 6.175863333	| 1.657173333 | 

#### CPU
This implementation does not involve the GPU at all and is purely single-threaded. This makes it faster for small arrays because no kernel launch is required. However, this algorithm scales poorly with array size compared to the other implementations because it does not take advantage of the multi-threaded approach that the other algorithms do. This approach faces bottlenecks in both memory and computation (it becomes slower as the array size gets large).

#### Naive
This algorithm requires two arrays that are swapped every iteration to avoid race conditions. Since it performs computations in parallel, it scales relatively well compared to the CPU approach. This algorithm is not as optimized as it could be; every iteration, all threads with index less than the stride value are idle. However, the most significant bottleneck comes from the kernel-launch overhead (there are log_2(n) kernels) the redundant computations where some threads re-add elements from the input array. 

#### Work-Efficient
The work-efficient algorithm uses up-sweep to build a sum tree and down-sweep to distribute prefix sums. Sine it opertes in-place, this saves memory. This approach also takes advantage of parallelism on the GPU and uses log_2(n) kernel launches, but each kernel does less extra work because there are fewer redundant computations. This approach still faces a bottleneck through the kernel-launch overhead (still log_2(n) kernels for upsweep and downsweep).

#### Thrust
The thrust approach is very fast on large arrays. It may be using shared memory or minimizing the number of idle threads to further optimize the algorithm.



### Example Output for Array Size 256
```
****************
** SCAN TESTS **
****************
    [  24   2  28  23  36  34  30  21  22  40  10   0  17 ...  20   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 0.0005ms    (std::chrono Measured)
    [   0  24  26  54  77 113 147 177 198 220 260 270 270 ... 6029 6049 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 0.0005ms    (std::chrono Measured)
    [   0  24  26  54  77 113 147 177 198 220 260 270 270 ... 5891 5934 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 0.823296ms    (CUDA Measured)
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 0.13824ms    (CUDA Measured)
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 0.635904ms    (CUDA Measured)
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 0.311296ms    (CUDA Measured)
    passed
==== thrust scan, power-of-two ====
   elapsed time: 0.203776ms    (CUDA Measured)
    passed
==== thrust scan, non-power-of-two ====
   elapsed time: 0.094208ms    (CUDA Measured)
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   0   2   0   3   0   2   2   1   0   0   2   2   1 ...   2   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 0.0008ms    (std::chrono Measured)
    [   2   3   2   2   1   2   2   1   3   1   2   3   3 ...   2   2 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 0.0004ms    (std::chrono Measured)
    [   2   3   2   2   1   2   2   1   3   1   2   3   3 ...   1   3 ]
    passed
==== cpu compact with scan ====
   elapsed time: 0.0048ms    (std::chrono Measured)
    [   2   3   2   2   1   2   2   1   3   1   2   3   3 ...   2   2 ]
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 0.538624ms    (CUDA Measured)
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 0.223232ms    (CUDA Measured)
    passed
```
