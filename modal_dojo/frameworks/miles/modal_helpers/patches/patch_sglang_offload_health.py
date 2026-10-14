"""Keep generation health checks out of offloaded GPU memory.

image: radixark/miles:dev-202610140350
sglang: f31278164939dd0746037abbb7566de4164e6767
file: python/sglang/srt/managers/scheduler.py

Miles can probe an idle engine after releasing its KV cache. SGLang must
acknowledge the health check without allocating tokens in unmapped memory.
TODO: remove when the pinned SGLang skips health generation while offloaded.
"""

from pathlib import Path

TARGET = Path("/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py")
MARKER = "PATCHED_MODAL_DOJO_OFFLOAD_HEALTH"
OLD = """            if is_health_check_generate_req(recv_req) and not self.is_fully_idle(
                for_health_check=True
            ):
"""
NEW = f"""            # {MARKER}: health probes must not touch offloaded memory.
            if is_health_check_generate_req(recv_req) and (
                self.weight_updater.offload_tags
                or not self.is_fully_idle(for_health_check=True)
            ):
"""


def patch_source(source: str) -> str:
    if MARKER in source:
        return source
    if source.count(OLD) != 1:
        raise ValueError("SGLang offload health patch anchor changed")
    source = source.replace(OLD, NEW, 1)
    compile(source, str(TARGET), "exec")
    return source


if __name__ == "__main__":
    TARGET.write_text(patch_source(TARGET.read_text()))
    print(f"Applied {MARKER} to {TARGET}")
