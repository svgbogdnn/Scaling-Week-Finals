import torch
import triton
import triton.language as tl
from torch.library import Library

try:
    _def_lib = Library("llm_scaling_week", "DEF")
    _def_lib.define("swiglu_fwd(Tensor a, Tensor b) -> Tensor")
except RuntimeError:
    pass


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
    ],
    key=["n_elements"],
)
@triton.jit
def swiglu_kernel(a_ptr, b_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    a_vals = tl.load(a_ptr + offs, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + offs, mask=mask, other=0.0)

    a_f32 = a_vals.to(tl.float32)
    b_f32 = b_vals.to(tl.float32)

    silu = a_f32 * tl.sigmoid(a_f32)
    out_f32 = silu * b_f32

    out_vals = out_f32.to(a_vals.dtype)
    tl.store(out_ptr + offs, out_vals, mask=mask)


def _swiglu_fwd_cuda(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if not (a.is_cuda and b.is_cuda):
        raise RuntimeError("swiglu_fwd: CUDA only")
    if a.dtype != b.dtype:
        raise RuntimeError("swiglu_fwd: dtypes must match")
    if a.shape != b.shape:
        raise RuntimeError("swiglu_fwd: shapes must match")
    if a.dtype not in (torch.float32, torch.bfloat16):
        raise RuntimeError("swiglu_fwd: supports only float32 and bfloat16")

    if not a.is_contiguous():
        a = a.contiguous()
    if not b.is_contiguous():
        b = b.contiguous()

    out = torch.empty_like(a)
    n_elements = a.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    swiglu_kernel[grid](a, b, out, n_elements)

    return out


def _swiglu_fwd_cpu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.silu(a) * b


_impl_lib = Library("llm_scaling_week", "IMPL")
_impl_lib.impl("swiglu_fwd", _swiglu_fwd_cuda, "CUDA")
_impl_lib.impl("swiglu_fwd", _swiglu_fwd_cpu, "CPU")