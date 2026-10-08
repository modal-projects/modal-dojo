"""Elide base64 blobs (audio/image data-URIs) from slime's logs.

Installs a ``logging.Handler`` wrapper via a site ``.pth`` so it loads in every
process — slime logs the prompt from the ``RolloutManager`` Ray actor, which a
driver-only hook misses.
"""

from __future__ import annotations

import os
import site
import sysconfig

# Raw string: the ``\1`` backref and ``\n`` must land verbatim in the written file.
HOOK_SOURCE = r"""import logging as _tg_logging
import re as _tg_re

if not getattr(_tg_logging.Handler, "_tg_base64_elided", False):
    _tg_rx = _tg_re.compile(r"(base64,)[A-Za-z0-9+/=]{64,}")
    _tg_orig_handle = _tg_logging.Handler.handle

    def _tg_handle(self, record):
        try:
            message = record.getMessage()
            if "base64," in message:
                record.msg = _tg_rx.sub(r"\1<elided>", message)
                record.args = None
        except Exception:
            pass
        return _tg_orig_handle(self, record)

    _tg_logging.Handler.handle = _tg_handle
    _tg_logging.Handler._tg_base64_elided = True
"""


def _site_dir() -> str:
    try:
        dirs = site.getsitepackages()
    except Exception:
        dirs = []
    if not dirs:
        dirs = [sysconfig.get_paths()["purelib"]]
    return dirs[0]


def main() -> None:
    target_dir = _site_dir()
    os.makedirs(target_dir, exist_ok=True)
    hook_path = os.path.join(target_dir, "_tg_base64_log_elide.py")
    pth_path = os.path.join(target_dir, "_tg_base64_log_elide.pth")
    with open(hook_path, "w") as f:
        f.write(HOOK_SOURCE)
    # A .pth line starting with ``import`` is executed by site.py at startup.
    with open(pth_path, "w") as f:
        f.write("import _tg_base64_log_elide\n")
    print(f"Installed base64 log-elide hook: {hook_path} (+ .pth)")


if __name__ == "__main__":
    main()
