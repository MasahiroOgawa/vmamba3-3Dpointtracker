"""Fall back to cuDNN-free convolutions when the installed cuDNN cannot finalize.

On this host a system-wide cuDNN 9.25 outranks the wheel's own 9.20 on the loader path and
supplies ``libcudnn_engines_tensor_ir``, which the wheel does not ship; the first convolution
then raises CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH. Only the DPT/depth convolutions are
affected and the non-cuDNN path costs nothing measurable at these sizes, so a run continues
rather than dying at step 0.

This is a workaround, not a fix. The fix is to remove the system cuDNN so the wheel's own is
used; the message says so rather than letting a silent fallback hide it. Mirrors
visionMamba3's depth/run.py::_survive_cudnn_mismatch so both repos behave the same.
"""

from __future__ import annotations

import torch


def survive_cudnn_mismatch() -> None:
    if not torch.cuda.is_available() or not torch.backends.cudnn.enabled:
        return
    try:
        torch.nn.functional.conv2d(
            torch.zeros(1, 1, 8, 8, device="cuda"), torch.zeros(1, 1, 3, 3, device="cuda")
        )
    except RuntimeError as e:
        if "CUDNN" not in str(e).upper():
            raise
        torch.backends.cudnn.enabled = False
        print(
            "[cudnn] unusable on this host (sublibrary version mismatch); running convolutions "
            "without it. Remove the system cuDNN to restore it.",
            flush=True,
        )
