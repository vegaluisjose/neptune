import torch

from matmul import matmul
from time import perf_counter


def print_capabilities():
    capability = torch.cuda.get_device_capability()
    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)

    print(f"CUDA Compute Capability: {capability}")
    print(f"Number of SMs: {props.multi_processor_count}")
    print(f"CUDA Capability Major/Minor version number: {props.major}.{props.minor}")


def init_input(m, n, dtype, acc_dtype):
    min_exp = -4 if acc_dtype == "float16" else -10
    exponents = torch.randint(min_exp, 0, size=(m, n))
    ret = (2.0**exponents).to(getattr(torch, dtype)).to("cuda")
    return ret


if __name__ == "__main__":
    torch.manual_seed(0)
    print_capabilities()

    M = 256
    N = 256
    K = 256
    dtype = "float16"
    a = init_input(M, K, dtype, None)
    b = init_input(K, N, dtype, None)

    tic = perf_counter()
    res = matmul(a, b, None, None, None)
    toc = perf_counter()

    print(f"Elapsed time (seconds): {toc-tic}")
