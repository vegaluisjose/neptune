import torch
import triton
import triton.language as tl

from time import perf_counter


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=8),
    ],
    key=["n_elements"],
)
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, output, n_elements)
    return output


if __name__ == "__main__":
    torch.manual_seed(0)
    size = 100000000
    x = torch.randn(size, device="cuda")
    y = torch.randn(size, device="cuda")
    tic = perf_counter()
    output = vector_add(x, y)
    toc = perf_counter()
    print(f"Elapsed time (seconds): {toc-tic}")
