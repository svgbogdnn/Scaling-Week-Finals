import triton
import torch

def pytorch_permute_index_map(tokens, indices):
    if indices.dim() == 1:
        topk = 1
    else:
        topk = indices.size(1)
    flatten_indices = indices.view(-1)
    sorted_indices = torch.argsort(flatten_indices, stable=True)
    num_out_tokens = flatten_indices.size(0)
    permuted_tokens = tokens.index_select(0, sorted_indices[:num_out_tokens] // topk)
    return permuted_tokens, sorted_indices


def torch_basic(x: torch.Tensor, top_experts: torch.Tensor, tokens_per_expert: torch.Tensor, topk: int, num_experts: int):
    block_size = 128
    device = x.device
    num_tokens, hidden_dim = x.shape

    expert_ids_flat = top_experts.view(-1)

    padded_tokens_per_expert = (
        ((tokens_per_expert + block_size - 1) // block_size) * block_size
    ).to(torch.int32)
    padded_offsets = torch.cat([
        torch.zeros(1, dtype=torch.int32, device=device),
        padded_tokens_per_expert.cumsum(dim=0)
    ])
    expert_ids_cpu = expert_ids_flat.cpu().tolist()
    padded_offsets_cpu = padded_offsets.cpu().tolist()

    max_padded_tokens = padded_offsets_cpu[-1]
    padded_tokens = torch.zeros(
        (max_padded_tokens, hidden_dim),
        dtype=x.dtype,
        device=device,
    )

    assignment_groups = [[] for _ in range(num_experts)]
    num_assignments = topk * num_tokens
    for i in range(num_assignments):
        expert_id = expert_ids_cpu[i]
        assignment_groups[expert_id].append(i)

    for e in range(num_experts):
        offset = padded_offsets[e]
        for local_idx, i in enumerate(assignment_groups[e]):
            original_token_idx = i // topk
            token_data = x[original_token_idx]
            target_row = offset + local_idx
            padded_tokens[target_row, :] = token_data

    return padded_tokens, padded_tokens_per_expert


def submission(
    x: torch.Tensor,
    top_experts: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    topk: int,
    num_experts: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor
]:
    device = x.device
    block_size = 128
    num_tokens, hidden_dim = x.shape

    tokens_per_expert_long = tokens_per_expert.to(device=device, dtype=torch.long)

    padded_tokens_per_expert_long = (
        (tokens_per_expert_long + (block_size - 1)) // block_size
    ) * block_size

    permuted_tokens, sorted_indices = pytorch_permute_index_map(x, top_experts)
    M = permuted_tokens.size(0)

    if M == 0:
        total_padded = int(padded_tokens_per_expert_long.sum().item())
        padded_tokens = torch.zeros(
            (total_padded, hidden_dim), dtype=x.dtype, device=device
        )
        return padded_tokens, padded_tokens_per_expert_long.to(torch.int32)

    flatten_expert_ids = top_experts.view(-1).to(device=device, dtype=torch.long)
    sorted_expert_ids = flatten_expert_ids.index_select(0, sorted_indices)

    if num_experts > 0:
        tpe_cumsum = tokens_per_expert_long.cumsum(dim=0)
        unpadded_offsets = tpe_cumsum - tokens_per_expert_long
    else:
        unpadded_offsets = torch.zeros(0, dtype=torch.long, device=device)

    if num_experts > 0:
        padded_cumsum = padded_tokens_per_expert_long.cumsum(dim=0)
        zero = torch.zeros(1, dtype=torch.long, device=device)
        padded_offsets_full = torch.cat([zero, padded_cumsum])
        padded_offsets_start = padded_offsets_full[:-1]
        total_padded = int(padded_offsets_full[-1].item())
    else:
        padded_offsets_start = torch.zeros(0, dtype=torch.long, device=device)
        total_padded = 0

    padded_tokens = torch.zeros(
        (total_padded, hidden_dim), dtype=x.dtype, device=device
    )

    if num_experts > 0:
        global_idx = torch.arange(M, device=device, dtype=torch.long)
        start_unpadded_for_assign = unpadded_offsets.index_select(0, sorted_expert_ids)
        pos_in_group = global_idx - start_unpadded_for_assign
        start_padded_for_assign = padded_offsets_start.index_select(0, sorted_expert_ids)
        dst_indices = start_padded_for_assign + pos_in_group
        padded_tokens.index_copy_(0, dst_indices, permuted_tokens)

    return padded_tokens, padded_tokens_per_expert_long.to(torch.int32)
