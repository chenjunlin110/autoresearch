#!/usr/bin/env python3

import json
import os
import socket
import sys

try:
    import torch
except Exception as exc:  # pragma: no cover
    torch = None
    torch_error = str(exc)
else:
    torch_error = None


def main() -> None:
    worker = sys.argv[1] if len(sys.argv) > 1 else "worker"
    data = {
        "worker": worker,
        "hostname": socket.gethostname(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_step_gpus": os.environ.get("SLURM_STEP_GPUS"),
    }
    if torch is None:
        data["torch_error"] = torch_error
    else:
        data["torch_cuda_device_count"] = torch.cuda.device_count()
        data["torch_current_device"] = torch.cuda.current_device() if torch.cuda.is_available() else -1
    print(json.dumps(data, sort_keys=True))


if __name__ == "__main__":
    main()
