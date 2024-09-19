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
Elapsed time (seconds): 0.45724202299606986
```

### Autotuning Vector addition

* Run
```bash
TRITON_PRINT_AUTOTUNING=1 python3 vadd_autotune/test.py
```

* Output
```bash
Triton autotuning for function add_kernel finished after 0.77s; best config selected: BLOCK_SIZE: 256, num_warps: 8, num_ctas: 1, num_stages: 2, maxnreg: None;
Elapsed time (seconds): 0.774298002012074
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
Triton autotuning for function _kernel finished after 3.21s; best config selected: BLOCK_M: 32, BLOCK_N: 32, BLOCK_K: 32, SPLIT_K: 1, num_warps: 2, num_ctas: 1, num_stages: 6, maxnreg: None;
Elapsed time (seconds): 3.4771489950071555
```
