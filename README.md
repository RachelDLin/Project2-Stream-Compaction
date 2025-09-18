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
