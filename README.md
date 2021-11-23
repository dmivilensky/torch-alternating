# `alternating` module

This module provides the extension for PyTorch toolkit, containing imlplementations of some alternating optimization methods
acting as envelopes for arbitrary optimizers applied to the problem considered. 

The usage example is provided in `alternating_demo.py`
for the Iris classification problem with the simplest multilayer neural network. 

The provided solution supports the following
alternating strategies:

* **Standard**. Blocks are switching in a subsequent way and optimization starts at the last obtained point
* **Accelerated**. Block are switching according to the specified policy, optimization starts at the point of one of the addtitional sequences

For the accelerated algorithm there are options for blocks switching policy:

* *Subsequent*. Blocks follow one after one circularly
* *Random*. Blocks are chosen randomly
* *MaxGrad*. Next block is one that has the greatest l2 norm of the corresponding gradient components
* *MinScalar*. Optimization performs for all the block one after one, but the resulting point is selected greedy by the smallness of loss value from the list of soltions, one per block
* *BestConvex*. The same procedure, but resulting point is chosen as optimal convex combination of the obtained solutions
