from types import SimpleNamespace
import pytest
import torch
from sglang.srt.mem_cache.index_key_cache import IndexKeyCache
from sglang.srt.layers.attention.dsa.hcu_int8_index_k_cache import create_index_k_int8_aliases
from sglang.kernels.ops.attention.dsa.index_buf_accessor import GetK, GetS


def make_cache(main_page):
    pool = SimpleNamespace(page_size=main_page, index_page_size=64,
        index_head_dim=128, quant_block_size=128, custom_mem_pool=None,
        indexer_layer_num=1, index_k_with_scale_buffer_dtype=torch.uint8,
        device="cpu", layer_transfer_counter=None, start_layer=0,
        _get_indexer_cache_index=lambda layer: 0)
    return pool, IndexKeyCache(pool, 1024)


@pytest.mark.parametrize("main_page", [64, 512])
def test_packed_index_pages_keep_physical_int8_abi(main_page):
    _, cache = make_cache(main_page)
    assert cache.buffer[0].shape == (17, 8448)
    keys, scales = create_index_k_int8_aliases(cache.buffer[0])
    assert keys.shape == (17, 64, 128)
    assert scales.shape == (17, 64)


@pytest.mark.parametrize("main_page", [64, 512])
def test_move_crosses_physical_index_pages(main_page):
    _, cache = make_cache(main_page)
    keys, scales = create_index_k_int8_aliases(cache.buffer[0])
    keys[0, 63].fill_(23)
    scales[0, 63] = 0.125
    cache.move(torch.tensor([512]), torch.tensor([63]))
    assert torch.all(keys[8, 0] == 23)
    assert scales[8, 0].item() == 0.125


@pytest.mark.parametrize("main_page", [64, 512])
def test_gather_uses_physical_index_pages(main_page):
    pool, cache = make_cache(main_page)
    keys, scales = create_index_k_int8_aliases(cache.buffer[0])
    keys[0].fill_(11)
    keys[1].fill_(22)
    scales[0].fill_(0.25)
    scales[1].fill_(0.5)
    pages = torch.tensor([1, 0], dtype=torch.int32)
    k = GetK.slow(pool, cache.buffer[0], 65, pages)
    s = GetS.slow(pool, cache.buffer[0], 65, pages).view(torch.float32).flatten()
    assert k.shape == (65, 128)
    assert torch.all(k[:64] == 22) and torch.all(k[64:] == 11)
    torch.testing.assert_close(s, torch.tensor([0.5] * 64 + [0.25]))


def test_shared_accessor_keeps_legacy_pool_page_size():
    pool, cache = make_cache(64)
    del pool.index_page_size
    keys, _ = create_index_k_int8_aliases(cache.buffer[0])
    keys[0].fill_(19)
    out = GetK.slow(pool, cache.buffer[0], 1, torch.tensor([0], dtype=torch.int32))
    assert torch.all(out == 19)


@pytest.mark.parametrize("main_page", [64, 512])
def test_indexer_native_int8_view_keeps_physical_pages(main_page):
    from sglang.srt.layers.attention.dsa.dsa_indexer import Indexer
    pool, cache = make_cache(main_page)
    indexer = object.__new__(Indexer)
    indexer.head_dim = 128
    indexer._get_index_k_read_buffer = lambda p, layer_id: cache.buffer[0]
    view = indexer._get_hcu_int8_paged_index_k_cache(pool, 0)
    assert view.shape == (17, 64, 1, 132)
    assert view.dtype == torch.int8
    assert view.data_ptr() == cache.buffer[0].data_ptr()
