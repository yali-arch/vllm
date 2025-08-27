from vllm.platforms import current_platform
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.attention.backends.utils import get_mla_dims
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE, cdiv
from tests.v1.attention.utils import (_Backend,
                                      create_common_attn_metadata,
                                      create_standard_kv_cache_spec,
                                      create_vllm_config,
                                      get_attention_backend,
                                      BatchSpec)
import triton
import torch
from itertools import product
import contextlib
import argparse
import os
import unittest.mock
import sys
sys.path.append(".")


BACKENDS_TO_TEST = [
    _Backend.CUTLASS_MLA,
    # _Backend.TRITON_MLA_VLLM_V1,
    # _Backend.FLASHMLA_VLLM_V1,
]

# Remove CUTLASS_MLA from the list if not using sm100
target_sm = torch.cuda.get_device_properties(0).major
if target_sm != 10:
    BACKENDS_TO_TEST.remove(_Backend.CUTLASS_MLA)
# if target_sm != 9:
#     BACKENDS_TO_TEST.remove(_Backend.FLASHMLA_VLLM_V1)

torch.manual_seed(42)

MODELS = (
    "/home/yali/scratch.yali_gpu/llm-models/DeepSeek-V3",
)

BATCHES = (
    ([64], [64]),
    # ([128], [128]),
    # ([1024], [1024]),
    # ([2048], [2048]),
    # ([32, 40], [32, 40]),
    # ([256, 512, 1024, 2048], [256, 512, 1024, 2048]),
    # ([512, 1024, 2048, 512, 1024, 2048], [512, 1024, 2048, 512, 1024, 2048]),
    # ([1024], [64]),
    # ([32, 40], [8, 8]),
    # ([256, 512, 1024, 2048], [16, 16, 16, 16]),
    # ([512, 1024, 2048, 512, 1024, 2048], [128, 256, 64, 512, 1024, 2048]),
    # ([64], [1]),
    # ([128], [1]),
    # ([1024], [1]),
    # ([2048], [1]),
    # ([32, 40], [1, 1]),
    # ([256, 512, 1024, 2048], [1, 1, 1, 1]),
    # ([512, 1024, 2048, 512, 1024, 2048], [1, 1, 1, 1, 1, 1]),
    # ([32, 40, 48, 56], [1, 1, 5, 5]),
    # ([256, 512, 1024, 2048], [1, 1, 16, 16]),
    # ([512, 1024, 2048, 512, 1024, 2048], [1, 1, 1, 7, 7, 7]),
)


MODELS_TO_BENCHMARK = list((m, *p) for (m, p) in product(MODELS, BATCHES))


def _convert_dtype_to_torch(dtype):
    """Convert ModelDType to torch.dtype."""
    if isinstance(dtype, str):
        if dtype == "auto":
            return torch.float16  # Default dtype for testing
        elif dtype in STR_DTYPE_TO_TORCH_DTYPE:
            return STR_DTYPE_TO_TORCH_DTYPE[dtype]
        else:
            raise ValueError(f"Unknown dtype: {dtype}")
    elif isinstance(dtype, torch.dtype):
        return dtype
    else:
        raise ValueError(f"Unknown dtype: {dtype}")


def create_dummy_kv_cache(kv_cache_spec: FullAttentionSpec,
                          device: torch.device,
                          num_blocks: int = 100) -> torch.Tensor:
    """Create a dummy KV cache tensor for testing."""
    kv_cache = torch.randn(
        num_blocks,
        kv_cache_spec.block_size,
        kv_cache_spec.head_size,  # latent dimension
        dtype=_convert_dtype_to_torch(kv_cache_spec.dtype),
        device=device,
    )
    return kv_cache


def create_and_prepopulate_kv_cache(
        kv_c_contexts: list[torch.Tensor],
        k_pe_contexts: list[torch.Tensor],
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        dtype: torch.dtype,
        device: torch.device,
        num_blocks: int,
        common_attn_metadata: CommonAttentionMetadata,
        randomize_blocks: bool = True) -> torch.Tensor:
    """Create and prepopulate an MLA KV cache with context data.

    Args:
        kv_c_contexts: List of latent KV context tensors for each sequence
        k_pe_contexts: List of key positional embedding context tensors
                       for each sequence
        block_size: Size of each block
        num_kv_heads: Number of KV heads (should be 1 for MLA)
        head_size: Size of each head (latent dimension)
        dtype: Data type for the cache
        device: Device to create the cache on
        num_blocks: Total number of blocks in the cache
        common_attn_metadata: Common attention metadata
        randomize_blocks: Whether to randomly permute blocks 
                          or use sequential order

    Returns:
        MLA KV cache tensor
    """
    batch_size = len(kv_c_contexts)
    seq_lens = common_attn_metadata.seq_lens_cpu
    query_lens = common_attn_metadata.query_start_loc_cpu[
        1:] - common_attn_metadata.query_start_loc_cpu[:-1]
    context_lens = common_attn_metadata.num_computed_tokens_cpu
    block_table = common_attn_metadata.block_table_tensor
    slot_mapping = common_attn_metadata.slot_mapping

    # Create MLA KV cache: (num_blocks, block_size, head_size)
    kv_cache = torch.empty(num_blocks,
                           block_size,
                           head_size,
                           dtype=dtype,
                           device=device)
    kv_cache_flat = kv_cache.view(-1, head_size)

    # Populate the cache with the context tokens
    # Start from block_id=1 since block_id=0 is considered the null block
    start_block_idx = 1
    for i in range(batch_size):
        kv_c_context, k_pe_context = kv_c_contexts[i], k_pe_contexts[i]
        kv_context = torch.cat([kv_c_context, k_pe_context.squeeze(1)], dim=-1)
        start = start_block_idx * block_size
        end = start + kv_context.shape[0]
        kv_cache_flat[start:end, ...] = kv_context

        # Stay block aligned and allocate enough blocks for the new tokens
        start_block_idx += cdiv(int(seq_lens[i]), block_size)

    blocks_end = start_block_idx

    # Permute the context blocks (excluding block 0 which is null)
    if randomize_blocks:
        perm = torch.randperm(
            blocks_end - 1) + 1  # Random permutation starting from block 1
    else:
        perm = torch.arange(
            1, blocks_end)  # Sequential order starting from block 1

    inv_perm = torch.zeros(blocks_end, dtype=torch.long, device=device)
    inv_perm[1:] = torch.argsort(
        perm) + 1  # Add 1 to account for starting from block 1
    kv_cache[1:blocks_end, ...] = kv_cache[perm, ...]

    # Construct the right block table
    # Start from block_id=1 since block_id=0 is considered the null block
    start_block_idx = 1
    for i in range(batch_size):
        num_blocks_for_seq = cdiv(int(seq_lens[i]), block_size)
        start = start_block_idx
        end = start + num_blocks_for_seq
        block_table[i, :num_blocks_for_seq] = inv_perm[start:end]
        start_block_idx += num_blocks_for_seq

        # Create a realistic slot mapping that corresponds to the block table
    for i in range(batch_size):
        token_offsets = torch.arange(int(query_lens[i])) + int(context_lens[i])
        block_indices = token_offsets // block_size
        token_inter_block_offsets = token_offsets % block_size
        start = common_attn_metadata.query_start_loc_cpu[i]
        end = common_attn_metadata.query_start_loc_cpu[i + 1]
        slot_mapping[start:end] = block_table[
            i,
            block_indices] * block_size + token_inter_block_offsets.to(device)

    return kv_cache


class MockAttentionLayer:
    """A mock attention layer for testing."""

    def __init__(self, device: torch.device):
        self._q_scale = torch.tensor(1.0, device=device)
        self._k_scale = torch.tensor(1.0, device=device)
        self._v_scale = torch.tensor(1.0, device=device)


def run_attention_backend(backend: _Backend, kv_cache_spec: FullAttentionSpec,
                          layer_names: list[str], vllm_config,
                          device: torch.device,
                          common_attn_metadata: CommonAttentionMetadata,
                          query: torch.Tensor, kv_c: torch.Tensor,
                          k_pe: torch.Tensor, kv_cache: torch.Tensor,
                          kv_lora_rank: int, qk_nope_head_dim: int,
                          qk_rope_head_dim: int, v_head_dim: int,
                          mock_kv_b_proj, use_cudagraph: bool) -> torch.Tensor:
    """Run attention computation using the specified backend's AttentionImpl."""

    builder_cls, impl_cls = get_attention_backend(backend)

    # Mock MLA's get_per_layer_parameters
    if vllm_config.model_config.use_mla:
        from vllm.v1.attention.backends.utils import PerLayerParameters

        def mock_get_per_layer_parameters(vllm_config, layer_names, impl_cls):
            # Return mock parameters for a single layer
            head_size = vllm_config.model_config.get_head_size()
            return {
                layer_name:
                PerLayerParameters(
                    window_left=-1,  # No sliding window
                    logits_soft_cap=0.0,  # No soft cap
                    sm_scale=1.0 / (head_size**0.5)  # Standard scale
                )
                for layer_name in layer_names
            }

        with unittest.mock.patch(
                'vllm.v1.attention.backends.mla.common.get_per_layer_parameters',
                mock_get_per_layer_parameters):
            builder = builder_cls(kv_cache_spec, layer_names, vllm_config,
                                  device)
            attn_metadata = builder.build(
                common_prefix_len=0,
                common_attn_metadata=common_attn_metadata,
            )
    else:
        # Build metadata
        builder = builder_cls(kv_cache_spec, layer_names, vllm_config, device)
        attn_metadata = builder.build(
            common_prefix_len=0,
            common_attn_metadata=common_attn_metadata,
        )

    # Instantiate implementation
    num_heads = vllm_config.model_config.get_num_attention_heads(
        vllm_config.parallel_config)
    num_kv_heads = vllm_config.model_config.get_num_kv_heads(
        vllm_config.parallel_config)
    head_size = vllm_config.model_config.get_head_size()
    scale = 1.0 / (head_size**0.5)
    impl = impl_cls(
        num_heads=num_heads,
        head_size=head_size,
        scale=scale,
        num_kv_heads=num_kv_heads,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
        logits_soft_cap=None,
        attn_type="decoder",
        kv_sharing_target_layer_name=None,
        q_lora_rank=None,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        qk_head_dim=qk_nope_head_dim + qk_rope_head_dim,
        v_head_dim=v_head_dim,
        kv_b_proj=mock_kv_b_proj,
    )

    # Process weights to create W_UK_T and W_UV attributes needed by MLA
    act_dtype = _convert_dtype_to_torch(vllm_config.model_config.dtype)
    impl.process_weights_after_loading(act_dtype)

    # Create mock layer and output buffer
    mock_layer = MockAttentionLayer(device)
    num_tokens = query.shape[0]
    output = torch.empty(num_tokens,
                         num_heads * v_head_dim,
                         dtype=query.dtype,
                         device=query.device)

    # Run forward pass
    # NOTE: The query, key, and value are already shaped correctly
    # in the calling test function.
    if use_cudagraph:
        times = triton.testing.do_bench_cudagraph(
            lambda: impl.forward(mock_layer,
                                 query,
                                 kv_c,
                                 k_pe,
                                 kv_cache,
                                 attn_metadata,
                                 output=output),
            return_mode="median")
    else:
        times = triton.testing.do_bench(
            lambda: impl.forward(mock_layer,
                                 query,
                                 kv_c,
                                 k_pe,
                                 kv_cache,
                                 attn_metadata,
                                 output=output),
            return_mode="median")
    return times, output


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["model", "seq_lens", "query_lens"],
        x_vals=MODELS_TO_BENCHMARK,
        line_arg="backend",
        line_vals=BACKENDS_TO_TEST,
        line_names=[b.name for b in BACKENDS_TO_TEST],
        ylabel="Runtime (ms)",
        plot_name="Benchmark MLA Backends with CUDA Graph",
        args={},
    )
)
def benchmark(backend, model, seq_lens, query_lens, use_cudagraph):
    print(
        f"Running benchmark with backend: {backend}, seq_lens: {seq_lens}, query_lens: {query_lens}, use_cudagraph: {use_cudagraph}")
    batch_spec = BatchSpec(seq_lens=seq_lens, query_lens=query_lens)
    vllm_config = create_vllm_config(model_name=model,
                                     max_model_len=max(seq_lens),
                                     num_gpu_blocks=2048)
    device = torch.device("cuda:0")

    kv_cache_spec = create_standard_kv_cache_spec(vllm_config)

    # 1. Setup
    mla_dims = get_mla_dims(vllm_config.model_config)
    batch_size = batch_spec.batch_size
    seq_lens = batch_spec.seq_lens
    query_lens = batch_spec.query_lens
    num_q_heads = vllm_config.model_config.get_num_attention_heads(
        vllm_config.parallel_config)
    num_kv_heads = vllm_config.model_config.get_num_kv_heads(
        vllm_config.parallel_config)
    head_size = vllm_config.model_config.get_head_size()
    dtype = _convert_dtype_to_torch(vllm_config.model_config.dtype)
    block_size = vllm_config.cache_config.block_size
    kv_lora_rank = mla_dims.kv_lora_rank
    qk_rope_head_dim = mla_dims.qk_rope_head_dim
    qk_nope_head_dim = mla_dims.qk_nope_head_dim
    v_head_dim = mla_dims.v_head_dim
    total_head_size = kv_lora_rank + qk_rope_head_dim
    assert kv_lora_rank + qk_rope_head_dim == head_size, \
        f"MLA dimensions don't match: {total_head_size} != {head_size}"
    scale = 1.0 / (total_head_size**0.5)

    # 2. Generate data and compute SDPA reference output
    all_q_vllm, all_kv_c_vllm, all_k_pe_vllm = [], [], []
    all_sdpa_outputs = []
    kv_c_contexts, k_pe_contexts = [], []

    # Create shared MLA weight matrices for consistency across all sequences
    W_UK = torch.randn(kv_lora_rank,
                       num_q_heads,
                       qk_nope_head_dim,
                       dtype=dtype,
                       device=device)
    W_UV = torch.randn(kv_lora_rank,
                       num_q_heads,
                       v_head_dim,
                       dtype=dtype,
                       device=device)
    kv_b_proj_weight = torch.cat([W_UK, W_UV], dim=-1)

    for i in range(batch_size):
        s_len = seq_lens[i]
        q_len = query_lens[i]
        context_len = s_len - q_len

        # Generate MLA tensors
        # Q has both nope and rope components:
        # [q_len, num_heads, qk_nope_head_dim + qk_rope_head_dim]
        q_c = torch.randn(q_len,
                          num_q_heads,
                          qk_nope_head_dim + qk_rope_head_dim,
                          dtype=dtype,
                          device=device)

        # KV_C (latent K/V): [s_len, kv_lora_rank]
        kv_c_full = torch.randn(s_len,
                                kv_lora_rank,
                                dtype=dtype,
                                device=device)

        # K_PE (rope component): [s_len, 1, qk_rope_head_dim]
        k_pe_full = torch.randn(s_len,
                                1,
                                qk_rope_head_dim,
                                dtype=dtype,
                                device=device)

        # Determine if this is decode (single token)
        # or prefill (multiple tokens)
        is_decode = q_len == 1

        # Split q into nope and rope components
        q_nope, q_pe = q_c.split([qk_nope_head_dim, qk_rope_head_dim], dim=-1)

        if is_decode:
            # Decode path: MQA-style attention in latent space
            # Transform q_nope to latent space: q_nope @ W_UK
            # q_nope: [1, num_heads, qk_nope_head_dim]
            # W_UK: [kv_lora_rank, num_heads, qk_nope_head_dim]
            ql_nope = torch.einsum("qnh,lnh->qnl", q_nope,
                                   W_UK)  # [1, num_heads, kv_lora_rank]

            # Build MQA attention inputs
            # Q: [1, num_heads, kv_lora_rank + qk_rope_head_dim]
            q_mqa = torch.cat([ql_nope, q_pe], dim=-1)
            # K: [s_len, kv_lora_rank + qk_rope_head_dim]
            # (broadcasted to all heads)
            k_mqa = torch.cat([kv_c_full, k_pe_full.squeeze(1)], dim=-1)
            k_mqa = k_mqa.unsqueeze(1).expand(-1, num_q_heads, -1)
            # V: [s_len, kv_lora_rank] (broadcasted to all heads)
            v_mqa = kv_c_full.unsqueeze(1).expand(-1, num_q_heads, -1)

            # SDPA expects (N, H, L, D)
            q_sdpa_in = q_mqa.unsqueeze(0).transpose(1, 2)
            k_sdpa_in = k_mqa.unsqueeze(0).transpose(1, 2)
            v_sdpa_in = v_mqa.unsqueeze(0).transpose(1, 2)

            sdpa_out_i = torch.nn.functional.scaled_dot_product_attention(
                q_sdpa_in, k_sdpa_in, v_sdpa_in, is_causal=False, scale=scale)
            sdpa_out_i = sdpa_out_i.transpose(1, 2).squeeze(
                0)  # [1, num_heads, kv_lora_rank]

            # Project back to output space: sdpa_out @ W_UV
            sdpa_out_i = torch.einsum("qnl,lnv->qnv", sdpa_out_i, W_UV)
            sdpa_out_i = sdpa_out_i.flatten(start_dim=-2)
        else:
            # Prefill path: MHA-style attention with full sequence
            # Apply kv_b_proj to the full kv_c tensor
            kv_nope_full = torch.einsum("sl,lnh->snh", kv_c_full,
                                        kv_b_proj_weight)
            k_nope_full, v_full = kv_nope_full.split(
                [qk_nope_head_dim, v_head_dim], dim=-1)

            # Build attention inputs for full sequence
            q_mha = torch.cat([q_nope, q_pe],
                              dim=-1)  # [q_len, num_heads, total_dim]
            k_pe_full_expanded = k_pe_full.expand(-1, num_q_heads, -1)
            k_full = torch.cat([k_nope_full, k_pe_full_expanded], dim=-1)

            # Create custom attention mask:
            # - Query tokens can attend to all context tokens
            # - Query tokens can only attend to query tokens up to their pos
            attn_mask = torch.ones(q_len,
                                   s_len,
                                   dtype=torch.bool,
                                   device=device)
            # Apply causal mask only to the query portion (context_len onwards)
            causal_mask = torch.tril(torch.ones(q_len, q_len, device=device))
            attn_mask[:, context_len:] = causal_mask

            # SDPA expects (N, H, L, D)
            q_sdpa_in = q_mha.unsqueeze(0).transpose(1, 2)
            k_sdpa_in = k_full.unsqueeze(0).transpose(1, 2)
            v_sdpa_in = v_full.unsqueeze(0).transpose(1, 2)

            # Single attention call with custom mask
            sdpa_out_i = torch.nn.functional.scaled_dot_product_attention(
                q_sdpa_in,
                k_sdpa_in,
                v_sdpa_in,
                attn_mask=attn_mask,
                scale=scale)
            sdpa_out_i = sdpa_out_i.transpose(1, 2).squeeze(0)
            sdpa_out_i = sdpa_out_i.flatten(start_dim=-2)

        all_sdpa_outputs.append(sdpa_out_i)

        # Inputs for vLLM MLA backends are just the new tokens
        all_q_vllm.append(q_c)
        all_kv_c_vllm.append(kv_c_full[context_len:])
        all_k_pe_vllm.append(k_pe_full[context_len:])

        # Contextual K/V data used to populate the paged cache (MLA format)
        kv_c_contexts.append(kv_c_full[:context_len])
        k_pe_contexts.append(k_pe_full[:context_len])

    # Concatenate all sequences (no reordering needed)
    query_vllm = torch.cat(all_q_vllm, dim=0)
    kv_c_vllm = torch.cat(all_kv_c_vllm, dim=0)
    k_pe_vllm = torch.cat(all_k_pe_vllm, dim=0)
    sdpa_output = torch.cat(all_sdpa_outputs, dim=0)

    # Create mock kv_b_proj using the same weights as reference implementation
    with (unittest.mock.patch(
            "vllm.model_executor.layers.linear.get_tensor_model_parallel_world_size",
            return_value=1),
          unittest.mock.patch(
              "vllm.model_executor.layers.linear.get_tensor_model_parallel_rank",
              return_value=0)):
        mock_kv_b_proj = ColumnParallelLinear(
            input_size=kv_lora_rank,
            output_size=num_q_heads *
            (qk_nope_head_dim + v_head_dim),
            bias=False).to(device=device,
                           dtype=dtype)

    # Set the mock weights to match our reference implementation
    # Reshape W_UK and W_UV to match the expected kv_b_proj format
    # [kv_lora_rank, num_heads, qk_nope_head_dim + v_head_dim]
    kv_b_proj_weight = kv_b_proj_weight.view(
        kv_lora_rank, num_q_heads * (qk_nope_head_dim + v_head_dim))
    mock_kv_b_proj.weight = torch.nn.Parameter(kv_b_proj_weight.T)

    # Create metadata using original batch spec
    common_attn_metadata = create_common_attn_metadata(
        batch_spec, vllm_config.cache_config.block_size, device)

    # 3. Simulate Paged KV Cache and a realistic slot_mapping
    kv_cache = create_and_prepopulate_kv_cache(
        kv_c_contexts=kv_c_contexts,
        k_pe_contexts=k_pe_contexts,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=dtype,
        device=device,
        num_blocks=vllm_config.cache_config.num_gpu_blocks,
        common_attn_metadata=common_attn_metadata,
        randomize_blocks=True)

    # 4. Run vLLM backends
    times, backend_output = run_attention_backend(backend, kv_cache_spec,
                                                  ["placeholder"], vllm_config,
                                                  device, common_attn_metadata,
                                                  query_vllm, kv_c_vllm,
                                                  k_pe_vllm,
                                                  kv_cache,
                                                  kv_lora_rank,
                                                  qk_nope_head_dim,
                                                  qk_rope_head_dim,
                                                  v_head_dim,
                                                  mock_kv_b_proj,
                                                  use_cudagraph)

    # Check shape and dtype consistency
    assert backend_output.shape == sdpa_output.shape, (
        f"[{backend}] shape {backend_output.shape} != "
        f"SDPA shape {sdpa_output.shape}")
    assert backend_output.dtype == sdpa_output.dtype, (
        f"[{backend}] dtype {backend_output.dtype} != "
        f"SDPA dtype {sdpa_output.dtype}")

    assert torch.isfinite(backend_output).all(), (
        f"[{backend}] produced non-finite values")

    # Check numerical similarity
    rtol = 1e-2
    atol = 5e-1

    max_diff = torch.max(torch.abs(backend_output - sdpa_output)).item()
    max_rel_diff = torch.max(
        torch.abs(backend_output - sdpa_output) /
        torch.abs(sdpa_output)).item()
    all_close = torch.allclose(backend_output,
                               sdpa_output,
                               rtol=rtol,
                               atol=atol)

    if not all_close:
        print(
            f"[{backend}] output differs from SDPA baseline. "
            f"Max diff: {max_diff:.6f}, max rel diff: {max_rel_diff:.6f})")

    return times


@contextlib.contextmanager
def temporary_environ(env_vars):
    """
    Temporarily set environment variables and restore them afterward.
    We have to do this vs monkeypatch because monkeypatch doesn't work
    with "module" scoped fixtures.
    """
    original_env = {k: os.environ.get(k) for k in env_vars}
    try:
        os.environ.update(env_vars)
        yield
    finally:
        for k, v in original_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda-graph", action="store_true",
                        help="Use CUDA graph for benchmarking")
    args = parser.parse_args()
    if args.cuda_graph:
        print("Using CUDA graph for benchmarking")

    env_vars = {
        "VLLM_USE_V1": "1",
        "VLLM_USE_CUDNN_PREFILL": "0",
    }

    with temporary_environ(env_vars):
        benchmark.run(
            print_data=True,
            show_plots=False,
            use_cudagraph=args.cuda_graph
        )
