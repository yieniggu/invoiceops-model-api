"""TEST-ONLY check child that skips structural validation.

It lets the tests prove that a crafted model which gets past validation (or a future bug in
the validator) still only crashes/hangs this disposable process, never the HTTP service.
Production entry points always validate (``python -m invoiceops_model_api._child``).
"""

from invoiceops_model_api import _child

_child.main(validate_structure=False)
