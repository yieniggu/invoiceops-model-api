"""TEST-ONLY inference worker whose NEXT model load is slow when SLOW_LOAD_FLAG exists.

The file holds the seconds to sleep; the worker deletes it as it starts sleeping, so exactly one
load (the first one after the test arms the flag) is slow. Used to prove that activating a model
does not block predictions served by other models.
"""

import os
import time

from invoiceops_model_api import _worker, model_runtime

FLAG = os.environ["SLOW_LOAD_FLAG"]
real_load = model_runtime.load_checked


def slow_load(path, max_uncompressed_bytes, **kwargs):
    try:
        with open(FLAG) as handle:
            seconds = float(handle.read().strip() or 0)
        os.unlink(FLAG)
    except (FileNotFoundError, ValueError):
        seconds = 0.0
    if seconds:
        time.sleep(seconds)
    return real_load(path, max_uncompressed_bytes, **kwargs)


model_runtime.load_checked = slow_load
_worker.main()
