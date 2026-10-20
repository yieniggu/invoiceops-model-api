"""Quiet spawn for CPython 3.12 on Linux/macOS, not a global start-method change.

The private multiprocessing preparation/reduction protocol matches 3.12's
popen_spawn_posix. Only process creation/reaping uses public subprocess APIs:
DEVNULL is installed before exec, including interpreter/bootstrap imports.
Revalidate this adapter before widening the project's Python/platform support.
"""

import io
import os
import subprocess
from multiprocessing import resource_tracker, spawn, util
from multiprocessing.context import SpawnContext, SpawnProcess, reduction, set_spawning_popen
from multiprocessing.popen_spawn_posix import Popen as SpawnPopen


class _QuietPopen(SpawnPopen):
    def _launch(self, process_obj):
        self.pid = None
        self._child = None
        tracker_fd = resource_tracker.getfd()
        self._fds.append(tracker_fd)
        preparation = spawn.get_preparation_data(process_obj._name)
        payload = io.BytesIO()
        set_spawning_popen(self)
        try:
            reduction.dump(preparation, payload)
            reduction.dump(process_obj, payload)
        finally:
            set_spawning_popen(None)

        parent_r = child_w = child_r = parent_w = None
        try:
            parent_r, child_w = os.pipe()
            self.sentinel = parent_r
            child_r, parent_w = os.pipe()
            # Keep the bootstrap writer as the parent's lifetime sentinel,
            # matching popen_spawn_posix (parent_process().is_alive()).
            self.finalizer = util.Finalize(self, util.close_fds, (parent_r, parent_w))
            command = spawn.get_command_line(tracker_fd=tracker_fd, pipe_handle=child_r)
            # Payload and trusted handles use anonymous descriptors, never argv,
            # files or a new environment channel. No output reader/buffer exists.
            self._child = subprocess.Popen(
                command,
                executable=spawn.get_executable(),
                pass_fds=tuple({*self._fds, child_r, child_w}),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self.pid = self._child.pid
            # Preserve ownership even if the subsequent bootstrap write fails:
            # BaseProcess.start has not assigned its returned Popen object yet.
            process_obj._popen = self
            process_obj._sentinel = self.sentinel
            with open(parent_w, "wb", closefd=False) as stream:
                stream.write(payload.getbuffer())
        finally:
            for fd in (child_r, child_w):
                if fd is not None:
                    os.close(fd)
            if self._child is None and parent_r is not None:
                if self.finalizer is not None:
                    self.finalizer()
                else:
                    os.close(parent_r)

    def poll(self, flag=os.WNOHANG):
        if self._child is not None:
            self.returncode = self._child.wait() if flag == 0 else self._child.poll()
        return self.returncode


class QuietSpawnProcess(SpawnProcess):
    @staticmethod
    def _Popen(process_obj):
        return _QuietPopen(process_obj)


class QuietSpawnContext(SpawnContext):
    Process = QuietSpawnProcess
