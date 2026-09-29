"""Replace the image's sglang source tree with the one radixark ships for V4.1.

Miles drives its engines through a weight-update session (``/begin_weight_update``
… ``/end_weight_update``) that exists only on sglang's ``sglang-miles`` branch, and
DeepSeek-V4.1 support exists only on sgl-project/sglang#38798 (the ``dsv4.1``
branch). Neither ref has the other, and the composition radixark trains V4.1 on
is not pushed to any public branch: it is published only as the source tree
inside ``radixark/miles:deepseek-v41`` (``dsv41.sglang-commit=4e7e72c50``),
which is an arm64-only image Modal cannot run. The tree itself is pure Python
plus JIT-compiled kernel sources, so pull that one layer straight from the
registry and unpack it over the amd64 nightly's ``sglang-miles`` checkout. The
layer is pinned by digest, and the nightly is the same ``lmsysorg/sglang:v0.5.18``
base that image was built on, so the compiled ``sgl_kernel`` matches.

Executed at image-build time via ``python3 <this file>``, before the patches
that edit files inside this tree.
"""

import hashlib
import io
import json
import pathlib
import shutil
import tarfile
import urllib.request

REPOSITORY = "radixark/miles"
# Layer `COPY sglang-python/python /sgl-workspace/sglang/python` of
# radixark/miles:deepseek-v41 (arm64 manifest, pushed 2026-09-10).
LAYER_DIGEST = "sha256:2dfcbad6ccc2e1e55ec69c50286d1e89343fb39f9cb944afa0b8f0be25e1adca"
LAYER_PREFIX = "sgl-workspace/sglang/python/"

SGLANG_PYTHON = pathlib.Path("/sgl-workspace/sglang/python")
MARKER = SGLANG_PYTHON / ".training_gym_dsv41_layer"

if MARKER.exists() and MARKER.read_text().strip() == LAYER_DIGEST:
    print("DeepSeek-V4.1 sglang tree already in place")
    raise SystemExit(0)

with urllib.request.urlopen(
    "https://auth.docker.io/token?service=registry.docker.io"
    f"&scope=repository:{REPOSITORY}:pull"
) as resp:
    token = json.load(resp)["token"]

req = urllib.request.Request(
    f"https://registry-1.docker.io/v2/{REPOSITORY}/blobs/{LAYER_DIGEST}",
    headers={"Authorization": f"Bearer {token}"},
)
with urllib.request.urlopen(req) as resp:
    blob = resp.read()

digest = "sha256:" + hashlib.sha256(blob).hexdigest()
if digest != LAYER_DIGEST:
    raise SystemExit(f"layer digest mismatch: got {digest}, want {LAYER_DIGEST}")

members = []
with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
    for member in tar.getmembers():
        if not member.name.startswith(LAYER_PREFIX):
            continue
        # AppleDouble sidecars from the machine the layer was assembled on.
        if pathlib.PurePosixPath(member.name).name.startswith("._"):
            continue
        if not (member.isfile() or member.isdir() or member.issym()):
            raise SystemExit(f"unexpected tar member {member.name} ({member.type!r})")
        member.name = member.name[len(LAYER_PREFIX) :]
        members.append(member)
    if not any(m.name == "sglang/srt/entrypoints/http_server.py" for m in members):
        raise SystemExit("layer does not look like an sglang python tree")
    if SGLANG_PYTHON.exists():
        shutil.rmtree(SGLANG_PYTHON)
    SGLANG_PYTHON.mkdir(parents=True)
    tar.extractall(SGLANG_PYTHON, members=members, filter="data")

MARKER.write_text(LAYER_DIGEST + "\n")
print(
    f"Unpacked {len(members)} entries from {REPOSITORY}@{LAYER_DIGEST} into {SGLANG_PYTHON}"
)
