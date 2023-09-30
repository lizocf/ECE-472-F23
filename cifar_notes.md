lzl notes:
 - MNIST model works incredibly slow with the cifar dataset, res. implementation will speed up this process
 - input & output layers must be same size for addition -> padding = SAME
 - data augmentation: try flipping 50%? 

# FIRST TRAINING ATTEMPT:
 - only tested on one batch
 - NO data augmentation implemented
 - NO checkpoints created .
 num_iters: 3000
 Loss => 0.9513286352157593, Accuracy => 79% (top_k_accuracy_score NOT implemented this run)
 num_params: 10212
``` learning:
   step_size: 0.05
   batch_size: 300
   num_iters: 3000
   decay_rate: 0.999
 data:
   num_samples: 800
   noise_stddev: 0.1
 conv:
   layer_depths: 32
   layer_kernel_sizes: 3
   num_conv_layers: 8
   num_res_blocks: [0,2,2,2,2]
 display:
   refresh_rate: 1
```

