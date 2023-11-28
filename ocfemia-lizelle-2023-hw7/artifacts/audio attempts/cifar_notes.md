lzl notes:
 - MNIST model works incredibly slow with the cifar dataset, res. implementation will speed up this process
 - input & output layers must be same size for addition -> padding = SAME
 - data augmentation: try flipping 50%? Unfortunately, I had to flip all input images because of RESOURCE_EXHAUSTED: failed to allocate memory


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

# SUCCESSFUL TRAINING ATTEMPT:
- tested with all batches.
- data augmentation implemented (flipped all images).
- NO checkpoints created.
- top_k_accuracy_score implemented.
- num_params: 
-----STEP_SIZE: 0.010700088274380404, BATCH_SIZE: 300, LAYER_DEPTH: 32 -----
Accuracy: 94%. Loss: 1.719804048538208. Steps Taken: 1540. 


# FIRST VALIDATION ATTEMPT:
-----STEP_SIZE: 0.009847429569214134, BATCH_SIZE: 300, LAYER_DEPTH: 32 -----
Accuracy: 96%. Loss: 1.9272381067276. Steps Taken: 1623. 

