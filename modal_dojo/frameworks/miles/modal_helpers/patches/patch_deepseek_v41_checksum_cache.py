"""Release checksum scratch memory before restoring DeepSeek's KV cache.

SGLang's checksum dequantizes FP8 weights into temporary BF16 tensors. PyTorch
keeps the freed allocations cached, but torch_memory_saver resumes the KV pool
through cuMemCreate, which cannot reclaim that cache. Miles calls onload_kv
immediately after the checksum, so return the scratch memory to CUDA first.

Image: radixark/miles:dsv41-h200-ea751aac8
SGLang: 7e74b31b2668934cf89dbe15188a010fda381aba
"""

from pathlib import Path

TARGET = Path("/sgl-workspace/sglang/python/sglang/srt/utils/weight_checker.py")
MARKER = "PATCHED_DEEPSEEK_V41_CHECKSUM_CACHE"
OLD = """        overall = overall_checksum(checksums)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
"""
NEW = f"""        overall = overall_checksum(checksums)

        torch.cuda.synchronize()
        # Release dequantization scratch before torch_memory_saver restores KV.
        torch.cuda.empty_cache()  # {MARKER}
        elapsed = time.perf_counter() - start
"""


if __name__ == "__main__":
    source = TARGET.read_text()
    if MARKER not in source:
        if source.count(OLD) != 1:
            raise ValueError("DeepSeek checksum cache patch did not match")
        TARGET.write_text(source.replace(OLD, NEW, 1))
    print("Patched checksum scratch cache release")
