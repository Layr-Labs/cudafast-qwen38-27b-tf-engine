#!/usr/bin/env bash
# test-no-keyring-exec.sh -- tools/no-keyring-exec.py, the guard that starts the
# participant adapter with the kernel keyring calls denied.
#
#   1. add_key, request_key and keyctl fail with EPERM under the guard (raw
#      syscalls, so no keyutils is needed), and work without it;
#   2. ordinary calls (file I/O, exec of a child) still work, the child
#      inherits the filter and no_new_privs, and it does not ignore SIGPIPE or
#      SIGXFSZ (Python ignores both; the fsize cap must still kill);
#   3. a wrong usage or a missing command exits 126 without running anything.
#   4. the first stderr line is the proof line the workflow counts.
#   5. the filter for BOTH tables (aarch64 and x86_64, whatever the host is), run
#      through a small BPF interpreter: the three keyring calls give EPERM,
#      other calls are allowed, a wrong audit arch kills, and x32 numbers kill;
#   6. the syscall numbers equal the kernel UAPI headers (asm-generic for
#      aarch64 always; x86_64 when that header is installed).
#
# Linux on aarch64 or x86_64 only (the platforms the guard supports).
set -uo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null && pwd -P)"
GUARD="${ROOT}/tools/no-keyring-exec.py"
failures=0
fail() { echo "FAIL: $*" >&2; failures=$((failures + 1)); }
ok()   { echo "ok: $*"; }

if [[ "$(uname -s)" != Linux ]]; then
  echo "test-no-keyring-exec.sh: needs Linux (seccomp); this host is $(uname -s)" >&2
  exit 1
fi

# probe -- try each keyring call once and print "name errno" lines.
probe() {
  python3 -I -S - <<'PY'
import ctypes, os, platform
nr = {"aarch64": (217, 218, 219), "x86_64": (248, 249, 250)}[platform.machine()]
libc = ctypes.CDLL(None, use_errno=True)
libc.syscall.restype = ctypes.c_long
calls = [
    ("add_key", nr[0], (b"user", b"guard-probe", b"v", 1, -4)),
    ("request_key", nr[1], (b"user", b"guard-probe-absent", None, 0)),
    ("keyctl", nr[2], (0, -4, 0, 0, 0)),  # KEYCTL_GET_KEYRING_ID @u
]
for name, number, args in calls:
    ctypes.set_errno(0)
    rc = libc.syscall(number, *args)
    print(name, 0 if rc >= 0 else ctypes.get_errno())
PY
}

guarded="$(python3 -I -S "${GUARD}" -- bash -c "$(declare -f probe); probe")"
for name in add_key request_key keyctl; do
  [[ "$(awk -v n="${name}" '$1 == n { print $2 }' <<< "${guarded}")" == 1 ]] \
    && ok "case 1: ${name} fails with EPERM under the guard" \
    || fail "case 1: ${name} under the guard: ${guarded}"
done
plain="$(probe)"
[[ "$(awk '$1 == "keyctl" { print $2 }' <<< "${plain}")" != 1 ]] \
  && ok "case 1: keyctl is not EPERM without the guard (the probe can tell)" \
  || fail "case 1: keyctl is EPERM even without the guard: ${plain}"

out="$(python3 -I -S "${GUARD}" -- bash -c 'cat /proc/self/status; head -c 1 /dev/zero | wc -c')"
grep -Eq '^NoNewPrivs:[[:space:]]+1$' <<< "${out}" && grep -Eq '^Seccomp:[[:space:]]+2$' <<< "${out}" \
    && grep -qx 1 <<< "${out}" \
  && ok "case 2: a child runs, reads files, and keeps no_new_privs and the filter" \
  || fail "case 2: unexpected child state: $(grep -E 'NoNewPrivs|Seccomp' <<< "${out}")"
# grep, not awk: awk ignores SIGPIPE by itself.
ignored="$(python3 -I -S "${GUARD}" -- grep '^SigIgn:' /proc/self/status | cut -f2)"
python3 -I -S - "${ignored}" <<'PY' \
  && ok "case 2: SIGPIPE and SIGXFSZ are not ignored in the child (the fsize cap still kills)" \
  || fail "case 2: the child ignores SIGPIPE or SIGXFSZ (SigIgn ${ignored})"
import signal, sys
mask = int(sys.argv[1], 16)
sys.exit(1 if mask & (1 << (signal.SIGPIPE - 1)) or mask & (1 << (signal.SIGXFSZ - 1)) else 0)
PY

marker="$(mktemp)"; rm -f "${marker}"
python3 -I -S "${GUARD}" touch "${marker}" >/dev/null 2>&1; status=$?
[[ "${status}" -eq 126 && ! -e "${marker}" ]] \
  && ok "case 3: a call without -- exits 126 and runs nothing" \
  || fail "case 3: no -- gave status ${status}"
python3 -I -S "${GUARD}" -- /nonexistent/adapter >/dev/null 2>&1; status=$?
[[ "${status}" -eq 126 ]] && ok "case 3: a missing command exits 126" \
  || fail "case 3: missing command gave status ${status}"
rm -f "${marker}"

first="$(python3 -I -S "${GUARD}" -- true 2>&1 >/dev/null | head -n 1)"
[[ "${first}" == "no-keyring-exec.py: guard active: Seccomp=2 NoNewPrivs=1" ]] \
  && ok "case 4: the first stderr line is the guard proof line" \
  || fail "case 4: first stderr line '${first}'"

python3 -I -S - "${GUARD}" <<'PY' && ok "case 5: both filters deny the keyring calls, allow others, kill a wrong arch and x32" \
  || fail "case 5: the generated filter is wrong (see above)"
import importlib.util, struct, sys
spec = importlib.util.spec_from_file_location("guard", sys.argv[1])
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)
DENY, ALLOW, KILL = g.SECCOMP_RET_ERRNO | g.EPERM, g.SECCOMP_RET_ALLOW, g.SECCOMP_RET_KILL_PROCESS


def run(code, count, nr, arch):
    ins = [struct.unpack("=HBBI", code[i * 8:i * 8 + 8]) for i in range(count)]
    data = {g.OFF_NR: nr, g.OFF_ARCH: arch}
    pc = acc = 0
    for _ in range(64):
        op, jt, jf, k = ins[pc]
        if op == g.LD_W_ABS:
            acc = data[k]
            pc += 1
        elif op == g.JEQ_K:
            pc += 1 + (jt if acc == k else jf)
        elif op == g.JGE_K:
            pc += 1 + (jt if acc >= k else jf)
        elif op == g.RET_K:
            return k
        else:
            raise SystemExit("unknown opcode %#x" % op)
    raise SystemExit("no return")


bad = []
for machine, (arch, *keyring) in g.ARCHES.items():
    x86 = machine == "x86_64"
    code, count = g.program(arch, *keyring, x86)
    other = [a for m, (a, *_) in g.ARCHES.items() if m != machine][0]
    cases = [(nr, arch, DENY) for nr in keyring]
    cases += [(nr, arch, ALLOW) for nr in (0, 1, 56, 63, 64, 93, 220, 221, 247, 251)
              if nr not in keyring]
    cases += [(keyring[0], other, KILL), (0, 0x40000003, KILL)]
    if x86:
        cases += [(nr | g.X32_SYSCALL_BIT, arch, KILL) for nr in keyring + [0, 1]]
    for nr, a, want in cases:
        got = run(code, count, nr, a)
        if got != want:
            bad.append("%s nr=%#x arch=%#x got %#x want %#x" % (machine, nr, a, got, want))
print("\n".join(bad))
sys.exit(1 if bad else 0)
PY

python3 -I -S - "${GUARD}" <<'PY' && ok "case 6: the syscall numbers equal the kernel UAPI headers" \
  || fail "case 6: a syscall number differs from the UAPI header, or the asm-generic header is missing"
import glob, importlib.util, re, sys
spec = importlib.util.spec_from_file_location("guard", sys.argv[1])
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)


def numbers(path):
    with open(path) as f:
        text = f.read()
    out = []
    for name in ("add_key", "request_key", "keyctl"):
        m = re.search(r"#define __NR_%s\s+(\d+)" % name, text)
        out.append(int(m.group(1)) if m else None)
    return out


bad = []
generic = "/usr/include/asm-generic/unistd.h"
try:
    if numbers(generic) != list(g.ARCHES["aarch64"][1:]):
        bad.append("aarch64 %s vs %s" % (numbers(generic), g.ARCHES["aarch64"][1:]))
except OSError:
    bad.append("missing " + generic)
for path in sorted(glob.glob("/usr/include/*/asm/unistd_64.h") + glob.glob("/usr/include/asm/unistd_64.h"))[:1]:
    if numbers(path) != list(g.ARCHES["x86_64"][1:]):
        bad.append("x86_64 %s: %s vs %s" % (path, numbers(path), g.ARCHES["x86_64"][1:]))
print("\n".join(bad))
sys.exit(1 if bad else 0)
PY

if [[ "${failures}" -eq 0 ]]; then
  echo "PASS: test-no-keyring-exec.sh"
  exit 0
fi
echo "FAIL: ${failures} failure(s)" >&2
exit 1
