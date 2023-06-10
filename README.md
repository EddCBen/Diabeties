## Architecture
- A Transformer-Encoder Layer, Model size = 20, number of heads = 1
-           paper: https://arxiv.org/abs/1706.03762
- Two Dense layers, the first expands the input into a 20 size 
- the second reduced the Trans-Enc output to a single float item


## Training:
- 5 Epochs
- Batch_size : 1

    ### Learning-Rates: 
    - 3.5e-5 for The transformer Encoder with a warmup phase: See the paper
    - 0.001 for the Dense layers also with a warmup stage

    ### Optimizers:
    -   AdamW for the Tansformer Encoder
    -    Adam for the Dense layers
