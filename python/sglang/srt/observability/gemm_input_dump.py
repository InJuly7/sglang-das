"""Metadata matching contiguous-GEMM debug logs, adapted to a masked API."""


def tensor_metadata(tensor):
    if tensor is None:
        return None
    return dict(
        shape=list(tensor.shape),
        dtype=str(tensor.dtype),
        device=str(tensor.device),
        stride=list(tensor.stride()),
        is_contiguous=tensor.is_contiguous(),
    )


def masked_gemm_metadata(
    api, layer, stage, a0, a1, b0, b1, output, masked_m, expected_m
):
    return dict(
        api=api,
        layer=layer,
        stage=stage,
        expected_m_per_group=expected_m,
        tensors={
            name: tensor_metadata(tensor)
            for name, tensor in (
                ("input", a0),
                ("a_scale", a1),
                ("b_qweight", b0),
                ("b_scale", b1),
                ("output", output),
                ("masked_m", masked_m),
            )
        },
        # Use the logical activation/output axes, never the packed W8/FP8
        # layout rules of another GEMM API to interpret an INT4 weight.
        derived_E=int(a0.shape[0]),
        derived_M=int(a0.shape[1]),
        derived_N=int(output.shape[-1]),
        derived_K=int(a0.shape[-1]),
    )


def masked_gemm_record(metadata, counts, *, run_id, rank, step, logical_ids):
    capacity = metadata["derived_M"]
    if len(counts) != metadata["derived_E"] or any(
        n < 0 or n > capacity for n in counts
    ):
        raise ValueError("Invalid masked_m values for GEMM capacity")
    valid = sum(counts)
    result = dict(metadata)
    result.update(
        schema_version=1,
        request_id=f"{run_id}:rank{rank}:step{step}:layer{metadata['layer']}:{metadata['stage']}",
        run_id=run_id,
        rank=rank,
        step=step,
        capacity_m_per_expert=capacity,
        total_m=len(counts) * capacity,
        valid_m=valid,
        pad_m=len(counts) * capacity - valid,
        masked_m_values=counts,
        expert_counts={str(i): n for i, n in enumerate(counts) if n > 0},
        physical_expert_ids=[rank * len(counts) + i for i in range(len(counts))],
        logical_expert_ids=logical_ids,
        # This API takes a per-expert count vector, not a per-row index tensor.
        m_indices_runs=None,
    )
    return result
