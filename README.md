# Group Normalization in Keras
A Keras implementation of [Group Normalization](https://arxiv.org/abs/1803.08494) by Yuxin Wu and Kaiming He.

Useful for fine-tuning of large models on smaller batch sizes than in research setting (where batch size is very large due to multiple GPUs). Similar to Batch Renormalization, but performs significantly better on ImageNet.

# Group Normalization

<img src="https://github.com/titu1994/Keras-Group-Normalization/blob/master/images/normalization_techniques.PNG?raw=true" height="100%" width="100%">

The above image is from the paper. It describes the differences between the 4 types of normalization techniques generally used. 

As can be seen, GN is independent of batchsize, which is crucial for fine-tuning large models which cannot be retrained with small batch sizes due to Batch Normalization's dependence on large batchsizes to compute the statistics of each batch and update its moving average perameters properly.

# Usage
Dropin replacement for BatchNormalization layers from Keras. The important parameter that is different from `BatchNormalization` is called `groups`. This must be appropriately set, and requires certain constraints such as :

1)  Needs to an integer by which the number of channels is divisible.
2)  `1 <= G <= #channels`, where #channels is the number of channels in the incomming layer.

```python
from group_norm import GroupNormalization

ip = Input(shape=(...))
x = GroupNormalization(groups=32, axis=-1)
...
```
