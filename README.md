# Neptune

## Getting started

* Install Triton, currently using `3.0.0`

```bash
python3 -m pip install triton
```

## Examples

### Vector addition

* Run
```bash
python3 vadd/test.py
```

* Output
```bash
tensor([1.3713, 1.3076, 0.4940,  ..., 0.6724, 1.2141, 0.9733], device='cuda:0')
tensor([1.3713, 1.3076, 0.4940,  ..., 0.6724, 1.2141, 0.9733], device='cuda:0')
The maximum difference between torch and triton is 0.0
```

### Autotuning Vector addition

* Run
```bash
TRITON_PRINT_AUTOTUNING=1 python3 vadd_autotune/test.py
```

* Output
```bash
Triton autotuning for function add_kernel finished after 0.48s; best config selected: BLOCK_SIZE: 128, num_warps: 4, num_ctas: 1, num_stages: 2, maxnreg: None;
tensor([ 1.3120, -0.5226,  2.0826,  ..., -0.4852,  1.7825, -1.1772],
       device='cuda:0')
```

### Autotuning Matrix Multiplication

* Run
```bash
TRITON_PRINT_AUTOTUNING=1 python3 matmul_autotune/test.py
```

* Output
```bash
CUDA Compute Capability: (8, 9)
Number of SMs: 128
CUDA Capability Major/Minor version number: 8.9
Triton autotuning for function _kernel finished after 2.28s; best config selected: BLOCK_M: 32, BLOCK_N: 32, BLOCK_K: 32, SPLIT_K: 1, num_warps: 2, num_ctas: 1, num_stages: 6, maxnreg: None;
```
