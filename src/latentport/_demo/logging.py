from contextlib import contextmanager
import os
import sys

@contextmanager
def quiet_native(log_path):
    # Windows ConsoleIO retains OS handles rather than following dup2(). Replace
    # the Python streams as well as the CRT descriptors, and update Win32's
    # standard handles (used by subprocesses and native libraries).
    original = (sys.stdout, sys.stderr)
    for stream in original:
        stream.flush()
    saved = [os.dup(fd) for fd in (1, 2)]
    console = os.fdopen(os.dup(saved[0]), "w", buffering=1,
                        encoding="utf-8", errors="replace")

    def sync_windows_handles():
        if os.name == "nt":
            import ctypes
            import msvcrt
            setter = ctypes.windll.kernel32.SetStdHandle
            setter.argtypes = [ctypes.c_ulong, ctypes.c_void_p]
            setter.restype = ctypes.c_int
            for number, fd in ((-11, 1), (-12, 2)):
                if not setter(number & 0xffffffff, msvcrt.get_osfhandle(fd)):
                    raise ctypes.WinError()

    redirected = []
    with log_path.open("w", encoding="utf-8") as log:
        try:
            os.dup2(log.fileno(), 1)
            os.dup2(log.fileno(), 2)
            sync_windows_handles()
            redirected = [os.fdopen(os.dup(fd), "w", buffering=1,
                          encoding="utf-8", errors="replace") for fd in (1, 2)]
            sys.stdout, sys.stderr = redirected
            yield console
        finally:
            for stream in redirected:
                stream.flush()
            for fd, backup in zip((1, 2), saved):
                os.dup2(backup, fd)
                os.close(backup)
            sync_windows_handles()
            if os.name == "nt":
                # The original ConsoleIO objects refer to handles closed by dup2.
                sys.stdout, sys.stderr = [os.fdopen(os.dup(fd), "w", buffering=1,
                    encoding="utf-8", errors="replace") for fd in (1, 2)]
            else:
                sys.stdout, sys.stderr = original
            for stream in redirected:
                stream.close()
            console.close()
