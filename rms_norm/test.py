import torch
import triton
import triton.language as tl

from time import perf_counter


@triton.jit
def op_kernel(
    x_ptr,
    w_ptr,
    output_ptr,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # block offsets
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    # ptr offsets
    x_ptr += block_start
    w_ptr += block_start
    output_ptr += block_start

    # accumulator
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # accumulate all pow(2) into acc
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + cols, mask=cols < N, other=0.0)
        acc += x * x

    # square root of average
    rms = tl.sqrt(tl.sum(acc) / N + eps)

    # compute reciprocal and scale with weights
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(w_ptr + cols, mask=mask)
        x = tl.load(x_ptr + cols, mask=mask, other=0.0)
        x_hat = x / rms
        y = x_hat * w

        tl.store(output_ptr + cols, y, mask=mask)


def op(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6):
    # define output
    output = torch.empty_like(x)

    # check all tensors in gpu
    assert x.is_cuda and w.is_cuda and output.is_cuda

    # get number of elements
    N = output.numel()

    # define grid
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)

    # launch kernel
    op_kernel[grid](x, w, output, N, eps, BLOCK_SIZE=256)

    return output


if __name__ == "__main__":
    torch.manual_seed(0)

    eps: float = 1e-6

    dim = 4096
    shape = [1, 1, dim]

    # inputs
    x = torch.rand(shape, device="cuda")
    w = torch.ones(dim, device="cuda")

    # run kernels
    tic = perf_counter()
    y = op(x, w, eps)
    toc = perf_counter()

    # compute reference
    x_hat = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    y_ref = x_hat * w

    print(f"Elapsed time (seconds): {toc-tic}")
    if torch.allclose(y, y_ref, atol=1e-2, rtol=0):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")
