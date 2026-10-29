#!/usr/bin/env python3
"""no-keyring-exec.py -- run a command with the kernel keyring syscalls denied.

TRUSTED: not in editablePaths. tools/run-engine-sandboxed.sh starts the
participant adapter through it. Kernel keys outlive the processes that make
them, and the sandbox user is shared by every submission, so a key could carry
state from one leg, job or participant to the next (docs/sandbox-design.md).
The resident and build containers already get Docker's default seccomp
profile, which denies these calls; this gives the host adapter the same rule.

Usage: no-keyring-exec.py -- COMMAND [ARGS...]

It sets no_new_privs, installs a seccomp filter that makes add_key,
request_key and keyctl fail with EPERM (every other call is allowed; a call
from another architecture or the x32 ABI kills the process), then execs
COMMAND. Linux on aarch64 or x86_64 only; anything else refuses to run.
Python ignores SIGPIPE and SIGXFSZ, and an ignored signal survives exec, so
both are reset to their defaults first: the adapter must still die at the
fsize cap and on a closed pipe, as it would without this helper.
Before the exec it writes one diagnostic line to stderr with its own Seccomp
and NoNewPrivs values. The job-log proof does not rely on it (the adapter can
rewrite its stderr file): tools/run-engine-sandboxed.sh reads the adapter's
/proc status itself.
"""
import ctypes
import os
import platform
import signal
import struct
import sys

# (AUDIT_ARCH_*, add_key, request_key, keyctl) from the kernel's syscall tables.
ARCHES = {
    "aarch64": (0xC00000B7, 217, 218, 219),
    "x86_64": (0xC000003E, 248, 249, 250),
}
PR_SET_NO_NEW_PRIVS = 38
PR_SET_SECCOMP = 22
SECCOMP_MODE_FILTER = 2
SECCOMP_RET_KILL_PROCESS = 0x80000000
SECCOMP_RET_ERRNO = 0x00050000
SECCOMP_RET_ALLOW = 0x7FFF0000
EPERM = 1
X32_SYSCALL_BIT = 0x40000000
# BPF opcodes: ld word abs, jeq k, jge k, ret k.
LD_W_ABS, JEQ_K, JGE_K, RET_K = 0x20, 0x15, 0x35, 0x06
OFF_NR, OFF_ARCH = 0, 4  # struct seccomp_data: int nr; __u32 arch; ...


def die(msg):
    sys.stderr.write("no-keyring-exec.py: %s\n" % msg)
    sys.exit(126)


def program(arch, add_key, request_key, keyctl, x86):
    deny = SECCOMP_RET_ERRNO | EPERM
    ins = [(LD_W_ABS, 0, 0, OFF_ARCH), None, (LD_W_ABS, 0, 0, OFF_NR)]
    if x86:
        ins.append(None)  # jge X32_SYSCALL_BIT -> kill
    ins += [(JEQ_K, 0, 1, add_key), (RET_K, 0, 0, deny),
            (JEQ_K, 0, 1, request_key), (RET_K, 0, 0, deny),
            (JEQ_K, 0, 1, keyctl), (RET_K, 0, 0, deny),
            (RET_K, 0, 0, SECCOMP_RET_ALLOW),
            (RET_K, 0, 0, SECCOMP_RET_KILL_PROCESS)]
    kill = len(ins) - 1
    ins[1] = (JEQ_K, 0, kill - 2, arch)
    if x86:
        ins[3] = (JGE_K, kill - 4, 0, X32_SYSCALL_BIT)
    return b"".join(struct.pack("=HBBI", *i) for i in ins), len(ins)


def main(argv):
    if len(argv) < 3 or argv[1] != "--":
        die("usage: no-keyring-exec.py -- COMMAND [ARGS...]")
    machine = platform.machine()
    if sys.platform != "linux" or machine not in ARCHES:
        die("unsupported platform %s/%s" % (sys.platform, machine))
    arch, add_key, request_key, keyctl = ARCHES[machine]
    code, count = program(arch, add_key, request_key, keyctl, machine == "x86_64")
    filters = ctypes.create_string_buffer(code, len(code))

    class SockFprog(ctypes.Structure):
        _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.c_void_p)]

    fprog = SockFprog(count, ctypes.addressof(filters))
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong,
                           ctypes.c_ulong, ctypes.c_ulong]
    if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        die("PR_SET_NO_NEW_PRIVS failed: %s" % os.strerror(ctypes.get_errno()))
    if libc.prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER,
                  ctypes.addressof(fprog), 0, 0) != 0:
        die("PR_SET_SECCOMP failed: %s" % os.strerror(ctypes.get_errno()))
    for sig in (signal.SIGPIPE, signal.SIGXFSZ):
        signal.signal(sig, signal.SIG_DFL)
    state = {}
    try:
        with open("/proc/self/status") as status:
            for line in status:
                key, _, value = line.partition(":")
                if key in ("Seccomp", "NoNewPrivs"):
                    state[key] = value.strip()
    except OSError:
        pass  # diagnostic only: the prctl results above are the gate
    sys.stderr.write("no-keyring-exec.py: guard active: Seccomp=%s NoNewPrivs=%s\n"
                     % (state.get("Seccomp", "?"), state.get("NoNewPrivs", "?")))
    sys.stderr.flush()
    try:
        os.execvp(argv[2], argv[2:])
    except OSError as err:
        die("cannot exec %s: %s" % (argv[2], err.strerror))


if __name__ == "__main__":
    main(sys.argv)
