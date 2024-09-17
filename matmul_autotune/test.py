import torch
from matmul import matmul, _matmul


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
    print_capabilities()

    M = 256
    N = 256
    K = 256
    dtype = "float16"
    a = init_input(M, K, dtype, None)
    b = init_input(K, N, dtype, None)
    res = matmul(a, b, None, None, None)
    exp = torch.matmul(a, b)

    torch.testing.assert_close(res, exp)
