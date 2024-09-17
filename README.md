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

### Vector addition with autotune

* Run
```bash
TRITON_PRINT_AUTOTUNING=1 python3 vadd_autotune/test.py
```

* Output
```
Triton autotuning for function add_kernel finished after 0.48s; best config selected: BLOCK_SIZE: 128, num_warps: 4, num_ctas: 1, num_stages: 2, maxnreg: None;
tensor([ 1.3120, -0.5226,  2.0826,  ..., -0.4852,  1.7825, -1.1772],
       device='cuda:0')
```
