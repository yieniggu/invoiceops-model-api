"""TEST-ONLY inference worker whose predict can be made to crash or hang on demand.

The mode is read from the file named by ``FAULT_FLAG_FILE`` on every predict: ``crash`` raises
SIGSEGV in this process (like a crafted tree array would), ``hang`` sleeps forever, anything
else behaves normally. Production runs ``python -m invoiceops_model_api._worker``.
"""

import os
import signal
import time

from invoiceops_model_api import _worker, model_runtime

FLAG = os.environ["FAULT_FLAG_FILE"]


def predict(model, matrix, columns, pos):
    try:
        with open(FLAG) as handle:
            mode = handle.read().strip()
    except FileNotFoundError:
        mode = ""
    if mode == "crash":
        os.kill(os.getpid(), signal.SIGSEGV)
        time.sleep(60)
    if mode == "hang":
        time.sleep(3600)
    return model_runtime.predict_positive(model, matrix, columns, pos)


_worker.main(predict_fn=predict)
