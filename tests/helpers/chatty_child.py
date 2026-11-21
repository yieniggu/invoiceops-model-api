"""TEST-ONLY check child that answers with whatever JSON the file named by CHATTY_REPLY_FILE holds.

It models a compromised child (a crafted model runs code in it): the parent must not forward
anything outside the fixed failure schema to the API caller.
"""

import os
import sys
from multiprocessing.connection import Connection

conn = Connection(int(sys.argv[1]))
conn.recv_bytes(64 * 1024)
with open(os.environ["CHATTY_REPLY_FILE"], "rb") as handle:
    conn.send_bytes(handle.read())
conn.close()
