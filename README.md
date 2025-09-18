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


#### Naive


#### Work-Efficient


#### Thrust



### Performance Bottlenecks



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
