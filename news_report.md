Rubric:
```
Picks a sensible approach for the task     [ ]
Meets core reporting requirements          [ ]
Meets core performance requirements [1]    [ ]
Avoids scope creep                         [ ]
Follow best practices for cross-validation [ ]
```
Pre-trained model used: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2


```
Final Validation Results: 
Loss: 0.04410463944077492, Accuracy: 98%

Test Results: 
Loss: 0.11611884087324142, Accuracy: 91%

Configs for both:
learning:
  step_size: 0.07
  batch_size: 500
  num_iters: 300
  decay_rate: 0.999
data:
  noise_stddev: 0.1
mlp:
  num_hidden_layers: 8
  hidden_layer_width: 30
display:
  refresh_rate: 1
```

Other notes:
- Adam optimizer and binary cross entropy used
- train/val/test split: 10000/2000/7600
- general structure: data -> transformer -> mlp 
