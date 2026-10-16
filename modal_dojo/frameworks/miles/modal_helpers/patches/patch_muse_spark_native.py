"""Repair the Muse vendor image native extensions for its pinned PyTorch 2.14.1.

The base ships PyTorch 2.14.1 but TE, FA2 and Apex binaries built for 2.13.
Rebuild only those extensions, without resolving dependencies or replacing TE's
CUDA core/Python patches. FA3 uses the stable ABI and remains unchanged.
"""

TE_SDIST = "https://files.pythonhosted.org/packages/9d/f1/ad613157261ac71262f5f260acf07fe22ae5ee0bb0675622c4f19a3e8afb/transformer_engine_torch-2.17.0.tar.gz#sha256=52ca109cdbc987ca02f87735fa375da8a02e117acb87e25dcb7e27dca37aa87f"


def rebuild_te():
    import importlib.metadata
    import os
    import subprocess
    import sys
    from pathlib import Path

    import torch

    assert str(torch.__version__) == "2.14.1+cu130"
    env = dict(
        os.environ,
        NVTE_PYTORCH_FORCE_BUILD="TRUE",
        NVTE_NO_LOCAL_VERSION="1",
        MAX_JOBS="16",
        TORCH_CUDA_ARCH_LIST="9.0;10.0;10.3",
    )
    cudnn = importlib.metadata.distribution("nvidia-cudnn-cu13")
    header = next(
        Path(cudnn.locate_file(p)) for p in cudnn.files if str(p).endswith("/cudnn.h")
    )
    env["CPATH"] = str(header.parent) + (":" + env["CPATH"] if env.get("CPATH") else "")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-v",
            "--no-cache-dir",
            "--no-deps",
            "--no-build-isolation",
            "--force-reinstall",
            TE_SDIST,
        ],
        env=env,
        check=True,
    )


def rebuild_fa2():
    import hashlib
    import os
    import subprocess
    import sys
    import tarfile
    import tempfile
    import urllib.request
    from pathlib import Path

    import torch

    assert str(torch.__version__) == "2.14.1+cu130"
    source = "https://files.pythonhosted.org/packages/11/34/9bf60e736ed7bbe15055ac2dab48ec67d9dbd088d2b4ae318fd77190ab4e/flash_attn-2.7.4.post1.tar.gz#sha256=f03485c9a49a4d68d0733acdcad80ab0e72afa025a777fdc2966ceccf9d51765"
    env = dict(
        os.environ,
        FLASH_ATTENTION_FORCE_BUILD="TRUE",
        FLASH_ATTN_CUDA_ARCHS="90;100",
        MAX_JOBS="16",
        NVCC_THREADS="2",
        TORCH_CUDA_ARCH_LIST="9.0;10.0;10.3",
    )
    with tempfile.TemporaryDirectory(prefix="muse-fa2-") as tmp:
        archive = Path(tmp) / "source.tar.gz"
        urllib.request.urlretrieve(source.split("#")[0], archive)
        assert (
            hashlib.sha256(archive.read_bytes()).hexdigest()
            == source.split("sha256=")[1]
        )
        with tarfile.open(archive) as tar:
            tar.extractall(tmp, filter="data")
        package = Path(tmp) / "flash_attn-2.7.4.post1"
        setup = package / "setup.py"
        text = setup.read_text()
        assert "-std=c++17" in text
        setup.write_text(text.replace("-std=c++17", "-std=c++20"))
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-v",
                "--no-cache-dir",
                "--no-deps",
                "--no-build-isolation",
                "--force-reinstall",
                str(package),
            ],
            env=env,
            check=True,
        )


def rebuild_apex():
    import os
    import subprocess
    import sys

    import torch

    assert str(torch.__version__) == "2.14.1+cu130"
    env = dict(
        os.environ,
        APEX_CPP_EXT="1",
        APEX_CUDA_EXT="1",
        MAX_JOBS="16",
        NVCC_APPEND_FLAGS="--threads 2",
        TORCH_CUDA_ARCH_LIST="9.0;10.0;10.3",
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-v",
            "--no-cache-dir",
            "--no-deps",
            "--no-build-isolation",
            "--force-reinstall",
            "git+https://github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4",
        ],
        env=env,
        check=True,
    )


def patch_te_fa4_gate():
    import importlib.metadata
    import importlib.util
    from pathlib import Path

    assert importlib.metadata.version("transformer-engine") == "2.17.0"
    spec = importlib.util.find_spec("transformer_engine")
    path = (
        Path(next(iter(spec.submodule_search_locations)))
        / "pytorch/attention/dot_product_attention/utils.py"
    )
    source = path.read_text()
    old = "    use_flash_attention_4 = use_flash_attention\n"
    new = (
        "    # Dojo Muse: FA4 backward is incompatible with pinned quack 0.6.5.\n"
        "    # Keep FA4 installed for SGLang; allow only TE callers to opt out.\n"
        '    use_flash_attention_4 = use_flash_attention and int(os.getenv("DOJO_MUSE_TE_FA4", "1"))\n'
    )
    if new not in source:
        assert source.count(old) == 1, "TE FA4 selection source changed"
        source = source.replace(old, new)
        source = source.replace(
            'logger.debug("Disabling FlashAttention 4 due to NVTE_FLASH_ATTN=0")',
            'logger.debug("Disabling FlashAttention 4 due to NVTE_FLASH_ATTN=0 or DOJO_MUSE_TE_FA4=0")',
        )
        compile(source, str(path), "exec")
        path.write_text(source)
    print("Dojo Muse TE FA4 gate installed:", path)
