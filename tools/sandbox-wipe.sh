#!/usr/bin/env bash
#
# sandbox-wipe.sh -- remove everything the sandbox user left behind, so nothing
# survives from one candidate leg or job into the next
# (see docs/sandbox-design.md, "Execution boundary"). TRUSTED: not in
# editablePaths. The workflow runs it at job start and in the cleanup step.
#
#   1. kills every sandbox process (nothing may recreate what is deleted);
#   2. as the sandbox user, deletes every file it owns under each directory in
#      the box's wipe list and deletes every child under its home (never
#      crossing a mount), after giving itself access to its own directories. The list
#      is every directory on the box the sandbox user can write (/tmp,
#      /var/tmp, /dev/shm, /run/lock, /var/crash, /dev/mqueue, ...), recorded
#      once by tools/box-runner/setup-sandbox-user.sh in a root-owned file. A
#      POSIX message queue is a file under /dev/mqueue, so it goes the same way;
#      a listed directory that does not exist (mqueue not mounted) is skipped;
#   3. as the sandbox user, removes every SysV shared memory segment, semaphore
#      set and message queue it owns, and empties its user, user-session and
#      persistent kernel keyrings (they outlive its processes; keyctl from
#      keyutils is required);
#   4. when SANDBOX_JOB is set, empties and replaces all five writable per-job
#      directories without following symlinks;
#   5. verifies nothing the sandbox user owns is left in any of those places,
#      and exits non-zero if something is. Only counts are printed: a name
#      could carry prompt text.
#
# A missing or empty wipe list is a failure: the box was never set up.
#
# Env: SANDBOX_USER (default qwen38sub) and SANDBOX_ETC (default
# /etc/qwen38-sandbox), the names tools/box-runner/setup-sandbox-user.sh
# uses. SANDBOX_WIPE_LIST (default $SANDBOX_ETC/wipe-dirs) and
# SANDBOX_WIPE_HOME (default /var/lib/$SANDBOX_USER) are for tests.
# No -e: every step's status is checked explicitly. A scan, chmod, delete, or
# root-reset error fails closed.
set -uo pipefail

# GOTCHA: candidate leg state has more than the box-wide wipe directories.
# These per-job paths are writable by the sandbox or resident container user
# and survive until the runner removes the job directory:
#   * zig-global-cache and zig-cache are writable build caches;
#   * out is the writable build output after its worker is staged elsewhere;
#   * adapter-home is the adapter HOME and TMPDIR;
#   * sock holds each socket directory and the resident run_home, which is the
#     resident HOME and TMPDIR.
# The wipe list and SANDBOX_WIPE_HOME cover shared temporary paths, POSIX IPC,
# and the sandbox home. SysV IPC is cleared separately below. The container's
# /tmp is a per-container tmpfs and does not survive docker run --rm.
# weights is runner-owned and read-only to the candidate. Runner-private logs
# are not readable by the candidate. Keep each fixed per-job directory, but
# replace it with the same owner, group, and mode after every candidate leg.

SANDBOX_USER="${SANDBOX_USER:-qwen38sub}"
SANDBOX_HOME="${SANDBOX_WIPE_HOME:-/var/lib/${SANDBOX_USER}}"
WIPE_LIST="${SANDBOX_WIPE_LIST:-${SANDBOX_ETC:-/etc/qwen38-sandbox}/wipe-dirs}"
[[ -f "${WIPE_LIST}" && -r "${WIPE_LIST}" ]] \
  || { echo "sandbox-wipe.sh: no wipe list ${WIPE_LIST}; run tools/box-runner/setup-sandbox-user.sh as root" >&2; exit 1; }
# One absolute path per line; anything else in the list is ignored.
DIRS=()
while IFS= read -r value; do DIRS+=("${value}"); done < <(grep '^/' "${WIPE_LIST}")
[[ "${#DIRS[@]}" -gt 0 ]] || { echo "sandbox-wipe.sh: the wipe list ${WIPE_LIST} names no directory" >&2; exit 1; }
DIRS+=("${SANDBOX_HOME}")
as_sandbox() { sudo -n -u "${SANDBOX_USER}" -- "$@"; }
stat_identity() {
  stat -c '%u %g %a' -- "$1" 2>/dev/null || stat -f '%u %g %Lp' -- "$1"
}
no_xattrs() {
  local status attrs
  if python3 - "$1" <<'PY'
import os
import sys

if not hasattr(os, "listxattr"):
    raise SystemExit(2)
try:
    attributes = os.listxattr(sys.argv[1])
except OSError:
    raise SystemExit(1)
raise SystemExit(0 if not attributes else 1)
PY
  then
    return 0
  else
    status=$?
  fi
  [[ "${status}" -eq 2 ]] || return 1
  if command -v getfattr >/dev/null 2>&1; then
    attrs="$(getfattr -d -m - -- "$1" 2>/dev/null)" || return 1
    ! grep -q '^[^#][^=]*=' <<< "${attrs}"
  elif command -v xattr >/dev/null 2>&1; then
    attrs="$(xattr "$1" 2>/dev/null)" || return 1
    if [[ "$(uname -s)" == Darwin ]]; then
      attrs="$(grep -v '^com\.apple\.provenance$' <<< "${attrs}")"
    fi
    [[ -z "${attrs}" ]]
  else
    return 1
  fi
}
clear_xattrs() {
  local status
  if python3 - "$1" <<'PY'
import os
import sys

if not hasattr(os, "listxattr") or not hasattr(os, "removexattr"):
    raise SystemExit(2)
try:
    for attribute in os.listxattr(sys.argv[1]):
        os.removexattr(sys.argv[1], attribute)
except OSError:
    raise SystemExit(1)
PY
  then
    return 0
  else
    status=$?
  fi
  [[ "${status}" -eq 2 ]] || return 1
  if command -v xattr >/dev/null 2>&1; then
    xattr -c "$1"
  else
    return 1
  fi
}
no_extended_acl() {
  local acl
  command -v getfacl >/dev/null 2>&1 || return 0
  acl="$(getfacl -cp -- "$1" 2>/dev/null)" || return 1
  ! grep -Eq '^(user|group):[^:]|^mask:|^default:' <<< "${acl}"
}
clear_extended_acl() {
  command -v setfacl >/dev/null 2>&1 || return 0
  setfacl -b -k -- "$1" >/dev/null 2>&1
}
root_is_clean() {
  local path="$1" identity="$2" found
  [[ -d "${path}" && ! -L "${path}" ]] || return 1
  [[ "$(stat_identity "${path}")" == "${identity}" ]] || return 1
  no_xattrs "${path}" || return 1
  no_extended_acl "${path}" || return 1
  found="$(find -P "${path}" -xdev -mindepth 1 -print -quit 2>/dev/null)" || return 1
  [[ -z "${found}" ]]
}

# wipe_root PATH ROOT_KIND -- remove and verify one root in one sandbox process.
# GOTCHA: a foreign private directory is safe to prune only when the sandbox
# user can neither write nor search it. A searchable or writable unreadable
# directory is a blind spot and fails the wipe.
wipe_root() {
  local result status=0
  # GOTCHA: -I -S: this python runs as the sandbox user with its HOME, so user site,
  # .pth files and PYTHON* variables are participant-controlled. Never load them.
  result="$(as_sandbox python3 -I -S - "$1" "${SANDBOX_UID}" "$2" "${SANDBOX_GID}" <<'PY'
import os
import shutil
import stat
import subprocess
import sys

root, sandbox_uid_text, root_kind, sandbox_gid_text = sys.argv[1:]
sandbox_uid = int(sandbox_uid_text)
sandbox_gid = int(sandbox_gid_text)
fixed_marker = ".qwen38-sandbox-wipe-marker"


def command(args):
    return subprocess.run(
        args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, check=False)


def identity(path):
    result = command(["stat", "-c", "%u %g %a", "--", path])
    if result.returncode:
        result = command(["stat", "-f", "%u %g %Lp", "--", path])
    if result.returncode:
        raise RuntimeError
    values = result.stdout.strip().split()
    if len(values) != 3 or not all(value.isdigit() for value in values):
        raise RuntimeError
    return int(values[0]), int(values[1]), int(values[2], 8)


def device(path):
    result = command(["stat", "-c", "%d", "--", path])
    if result.returncode:
        result = command(["stat", "-f", "%d", "--", path])
    if result.returncode or not result.stdout.strip().isdigit():
        raise RuntimeError
    return int(result.stdout.strip())


def can(flag, path):
    return command(["test", flag, path]).returncode == 0


def walk(path, remove_all, delete, root_device):
    os.lstat(path)
    uid, _gid, _mode = identity(path)
    if device(path) != root_device:
        return
    if uid == sandbox_uid:
        os.chmod(path, os.lstat(path).st_mode | stat.S_IRWXU)
    else:
        writable = can("-w", path)
        searchable = can("-x", path)
        if not writable and not searchable:
            return
        # GOTCHA: on Linux a writer of a non-sticky directory may set user.*
        # xattrs on it; a writable foreign directory without search access
        # also escapes the walk. Both carry state: fail closed.
        if writable and (not searchable
                         or not os.lstat(path).st_mode & stat.S_ISVTX):
            raise RuntimeError
        if not can("-r", path):
            raise RuntimeError

    with os.scandir(path) as entries:
        children = [entry.path for entry in entries]
    # GOTCHA: programs of other users create and remove temp files during the
    # walk. A child that is gone when a step fails holds no state: skip it.
    # Sandbox processes are dead, so nothing of the sandbox can vanish alone.
    for child in children:
        try:
            visit(child, remove_all, delete, root_device)
        except Exception:
            if os.path.lexists(child):
                raise


def visit(child, remove_all, delete, root_device):
        before = identity(child)
        child_stat = os.lstat(child)
        child_owned = before[0] == sandbox_uid
        # GOTCHA: a regular file of another user that the sandbox user can
        # write is a channel (bytes, xattrs, hardlink names, preallocated
        # blocks). Fail closed at any size. The box must not offer one: a
        # shared lock must deny the sandbox user (see the box setup notes).
        if (stat.S_ISREG(child_stat.st_mode) and not child_owned
                and can("-w", child)):
            raise RuntimeError
        if stat.S_ISDIR(child_stat.st_mode) and not stat.S_ISLNK(child_stat.st_mode):
            if device(child) != root_device:
                return
            walk(child, remove_all, delete, root_device)
            if remove_all or child_owned:
                if not delete:
                    raise RuntimeError
                after = identity(child)
                if after[:2] != before[:2]:
                    raise RuntimeError
                os.rmdir(child)
        elif remove_all or child_owned:
            if not delete or identity(child) != before:
                raise RuntimeError
            os.unlink(child)


def clear_xattrs(path):
    if hasattr(os, "listxattr") and hasattr(os, "removexattr"):
        for attribute in os.listxattr(path):
            if sys.platform == "darwin" and attribute == "com.apple.provenance":
                continue
            os.removexattr(path, attribute)
        return
    if command(["xattr", "-c", path]).returncode:
        raise RuntimeError


def check_xattrs(path):
    if hasattr(os, "listxattr"):
        attributes = os.listxattr(path)
        if sys.platform == "darwin":
            attributes = [name for name in attributes
                          if name != "com.apple.provenance"]
        if attributes:
            raise RuntimeError
        return
    result = command(["xattr", path])
    if result.returncode or result.stdout.strip():
        raise RuntimeError


def clear_acl(path):
    if not shutil.which("getfacl"):
        return
    if not shutil.which("setfacl"):
        raise RuntimeError
    if command(["setfacl", "-b", "-k", "--", path]).returncode:
        raise RuntimeError


def check_acl(path):
    if not shutil.which("getfacl"):
        return
    result = command(["getfacl", "-cp", "--", path])
    if result.returncode:
        raise RuntimeError
    for line in result.stdout.splitlines():
        fields = line.split(":")
        if (line.startswith("mask:") or line.startswith("default:")
                or fields[0] in ("user", "group")
                and len(fields) > 1 and fields[1]):
            raise RuntimeError


# GOTCHA: inode flags the owner may set (ext4 nodump, noatime, ...; BSD UF_*) carry
# bits across legs. Clear them as the sandbox user, then verify.
USER_INODE_FLAGS = (0x1 | 0x2 | 0x4 | 0x8 | 0x40 | 0x80 | 0x400 | 0x8000 | 0x10000 | 0x20000
                    | 0x800000 | 0x02000000 | 0x20000000 | 0x40000000)


def clear_inode_flags(path, verify=False):
    if sys.platform == "darwin":
        flags = os.stat(path, follow_symlinks=False).st_flags & 0xFFFF
        if flags and not verify:
            os.chflags(path, 0, follow_symlinks=False)
        elif flags:
            raise RuntimeError
        return
    import array
    import errno
    import fcntl
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        buf = array.array("l", [0])
        try:
            fcntl.ioctl(fd, 0x80086601, buf, True)
        except OSError as error:
            if error.errno in (errno.ENOTTY, errno.EOPNOTSUPP):
                return
            raise
        if buf[0] & USER_INODE_FLAGS and verify:
            raise RuntimeError
        if buf[0] & USER_INODE_FLAGS:
            buf[0] &= ~USER_INODE_FLAGS
            fcntl.ioctl(fd, 0x40086602, buf)
    finally:
        os.close(fd)


def main():
    if root_kind not in ("root", "home"):
        raise RuntimeError
    if root_kind == "root" and not os.path.lexists(root):
        return "SKIP"
    if not root.startswith("/") or root == "/" or os.path.islink(root):
        raise RuntimeError
    if not os.path.isdir(root):
        raise RuntimeError
    root_identity = identity(root)
    owned_root = root_identity[0] == sandbox_uid
    if root_kind == "home" and not owned_root:
        raise RuntimeError
    root_device = device(root)
    walk(root, root_kind == "home", True, root_device)
    if owned_root:
        clear_xattrs(root)
        clear_acl(root)
        clear_inode_flags(root)
        if root_kind == "home":
            os.chown(root, -1, sandbox_gid, follow_symlinks=False)
            os.chmod(root, 0o700)
            if identity(root) != (root_identity[0], sandbox_gid, 0o700):
                raise RuntimeError
        check_xattrs(root)
        check_acl(root)
        clear_inode_flags(root, verify=True)
    walk(root, root_kind == "home", False, root_device)
    if root_kind == "home":
        os.utime(root, ns=(0, 0), follow_symlinks=False)
        root_stat = os.stat(root, follow_symlinks=False)
        if root_stat.st_atime_ns != 0 or root_stat.st_mtime_ns != 0:
            raise RuntimeError
    elif not owned_root and can("-w", root) and can("-x", root):
        # A root the sandbox user cannot write or search holds no
        # participant timestamp state, and a marker there would fail.
        marker = os.path.join(root, fixed_marker)
        descriptor = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(descriptor)
        os.unlink(marker)
    return "PASS"


try:
    outcome = main()
except Exception:
    print("sandbox-wipe-root: FAIL")
    raise SystemExit(1)
print("sandbox-wipe-root: " + outcome)
PY
)" || status=$?
  [[ "${status}" -eq 0 \
      && ( "${result}" == "sandbox-wipe-root: PASS" \
        || "${result}" == "sandbox-wipe-root: SKIP" ) ]]
}

clear_leg_dir() {
  local path="$1" identity uid gid mode found
  # A per-job dir that does not exist yet (sandbox start) holds no state.
  [[ -e "${path}" || -L "${path}" ]] || return 0
  [[ -d "${path}" && ! -L "${path}" ]] || return 1
  identity="$(stat_identity "${path}")" || return 1
  read -r uid gid mode <<< "${identity}"
  [[ "${uid}" =~ ^[0-9]+$ && "${gid}" =~ ^[0-9]+$ && "${mode}" =~ ^[0-7]+$ ]] || return 1

  # -P and -xdev keep planted links and mounts from changing the wipe target.
  as_sandbox find -P "${path}" -xdev -mindepth 1 -type d -user "${SANDBOX_USER}" \
    -exec chmod u+rwx {} \; >/dev/null 2>&1 || return 1
  as_sandbox find -P "${path}" -xdev -mindepth 1 -depth -user "${SANDBOX_USER}" \
    -delete >/dev/null 2>&1 || return 1
  find -P "${path}" -xdev -mindepth 1 -depth -delete >/dev/null 2>&1 || return 1
  found="$(find -P "${path}" -xdev -mindepth 1 -print -quit 2>/dev/null)" || return 1
  [[ -z "${found}" ]] || return 1

  rmdir -- "${path}" || return 1
  mkdir -m "${mode}" -- "${path}" || return 1
  chown "${uid}:${gid}" "${path}" || return 1
  chmod "${mode}" "${path}" || return 1
  clear_xattrs "${path}" || return 1
  clear_extended_acl "${path}" || return 1
  root_is_clean "${path}" "${identity}"
}

as_sandbox true || { echo "sandbox-wipe.sh: 'sudo -n -u ${SANDBOX_USER}' fails; cannot wipe" >&2; exit 1; }
SANDBOX_UID="$(id -u "${SANDBOX_USER}" 2>/dev/null)" \
  || { echo "sandbox-wipe: FAIL: leg-state-not-cleared" >&2; exit 1; }
SANDBOX_GID="$(id -g "${SANDBOX_USER}" 2>/dev/null)" \
  || { echo "sandbox-wipe: FAIL: leg-state-not-cleared" >&2; exit 1; }
as_sandbox kill -KILL -1 >/dev/null 2>&1

# ipc_ids FLAG -- ids of the SysV objects of one kind (-m, -s, -q) the sandbox
# user created OR owns. An owner can hand an object to another uid (IPC_SET)
# and, as its creator, keep using it, so the creator column counts too; the
# creator may also remove it. ipcs -c prints: id perms cuid cgid uid gid
# (a name, or the number for a uid with no name).
ipc_ids() {
  ipcs "$1" -c 2>/dev/null | awk -v n="${SANDBOX_USER}" -v u="${SANDBOX_UID}" \
    '$1 ~ /^[0-9]+$/ && ($3 == n || $3 == u || $5 == n || $5 == u) { print $1 }'
}

# ipc_foreign_open FLAG -- ids of SysV objects of one kind that another uid
# created and owns, but the sandbox user may read or write: a world bit, or a
# group bit when the object's creator group or group is one of its groups (Linux
# checks both; by name or gid). The sandbox user cannot remove them and could
# keep state in them, so they fail the wipe. A permission digit of 2 or more has
# the read or the write bit. A failed group lookup fails the wipe (fail closed).
sandbox_gids="$(id -G "${SANDBOX_USER}" 2>/dev/null)" && [[ -n "${sandbox_gids}" ]] \
  || { echo "sandbox-wipe: FAIL: leg-state-not-cleared" >&2; exit 1; }
sandbox_gnames="$(id -Gn "${SANDBOX_USER}" 2>/dev/null)" && [[ -n "${sandbox_gnames}" ]] \
  || { echo "sandbox-wipe: FAIL: leg-state-not-cleared" >&2; exit 1; }
SANDBOX_GROUPS=" ${sandbox_gids} ${sandbox_gnames} "
ipc_foreign_open() {
  ipcs "$1" -c 2>/dev/null | awk -v n="${SANDBOX_USER}" -v u="${SANDBOX_UID}" -v g="${SANDBOX_GROUPS}" \
    '$1 ~ /^[0-9]+$/ && !($3 == n || $3 == u || $5 == n || $5 == u) {
       p = $2 % 1000; other = p % 10; grp = int(p / 10) % 10
       if (other >= 2 || (grp >= 2 && (index(g, " " $4 " ") > 0 || index(g, " " $6 " ") > 0))) print $1
     }'
}

# clear_keyrings -- empty the sandbox user's user (@u), user-session (@us) and,
# when the kernel has one, persistent keyring, then check that each is empty.
# Keys outlive the processes that made them, so they would carry state from one
# leg, job or participant to the next. One sandbox process does it all, in a new
# anonymous session keyring (sudo here runs no pam_keyinit, so the inherited
# session keyring would be the runner's own): linking @u into that session
# keyring makes @u and the persistent keyring (linked into @u) possessed, which
# clearing the persistent keyring needs; the session keyring is cleared and
# checked last. A get_persistent failure other than "not supported" fails the
# wipe. Only a result is printed: a key description could carry prompt text.
clear_keyrings() {
  command -v keyctl >/dev/null 2>&1 || {
    echo "sandbox-wipe.sh: keyctl (keyutils) is required to clear the ${SANDBOX_USER} keyrings" >&2
    return 1
  }
  # shellcheck disable=SC2016 # expanded by the sandbox user's bash, not here
  if ! as_sandbox keyctl session - bash -c '
      export LC_ALL=C
      keyctl link @u @s >/dev/null 2>&1 || exit 1
      rings="@u @us"
      rc=0
      if p="$(keyctl get_persistent @u 2>&1)"; then
        if [[ "${p}" =~ ^[0-9]+$ ]]; then rings="${p} ${rings}"; else rc=1; fi
      elif [[ "${p}" != *"Operation not supported"* ]]; then
        rc=1
      fi
      empty() { local out; out="$(keyctl rlist "$1" 2>/dev/null)" && [[ -z "${out//[[:space:]]/}" ]]; }
      for r in ${rings}; do keyctl clear "${r}" >/dev/null 2>&1 || rc=1; done
      for r in ${rings}; do empty "${r}" || rc=1; done
      keyctl clear @s >/dev/null 2>&1 || rc=1
      empty @s || rc=1
      exit "${rc}"' >/dev/null 2>&1; then
    echo "sandbox-wipe.sh: the ${SANDBOX_USER} kernel keyrings are not empty" >&2
    return 1
  fi
}

leg_state_failed=0
for d in "${DIRS[@]}"; do
  [[ "${d}" == "${SANDBOX_HOME}" ]] && continue
  wipe_root "${d}" root || leg_state_failed=1
done
for kind in -m -s -q; do
  ids="$(ipc_ids "${kind}")" || { leg_state_failed=1; continue; }
  for id in ${ids}; do
    as_sandbox ipcrm "${kind}" "${id}" >/dev/null 2>&1 || leg_state_failed=1
  done
done
clear_keyrings || leg_state_failed=1

wipe_root "${SANDBOX_HOME}" home || leg_state_failed=1

if [[ -n "${SANDBOX_JOB:-}" ]]; then
  if [[ "${SANDBOX_ROOT:-}" != /* \
        || ! "${SANDBOX_JOB:-}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]]; then
    leg_state_failed=1
  else
    job_dir="${SANDBOX_ROOT}/${SANDBOX_JOB}"
    for name in zig-global-cache zig-cache out adapter-home sock; do
      clear_leg_dir "${job_dir}/${name}" || leg_state_failed=1
    done
  fi
fi

left=0
for kind in -m -s -q; do
  ids="$(ipc_ids "${kind}")" || { left=1; continue; }
  if [[ -n "${ids}" ]]; then
    n="$(printf '%s\n' "${ids}" | wc -l | tr -d '[:space:]')"
    echo "sandbox-wipe.sh: ${n} SysV IPC object(s) created or owned by ${SANDBOX_USER} (ipcs ${kind}) remain" >&2
    left=1
  fi
  ids="$(ipc_foreign_open "${kind}")" || { left=1; continue; }
  if [[ -n "${ids}" ]]; then
    n="$(printf '%s\n' "${ids}" | wc -l | tr -d '[:space:]')"
    echo "sandbox-wipe.sh: ${n} SysV IPC object(s) of another user that ${SANDBOX_USER} can read or write (ipcs ${kind})" >&2
    left=1
  fi
done
if [[ "${left}" -ne 0 || "${leg_state_failed}" -ne 0 ]]; then
  echo "sandbox-wipe: FAIL: leg-state-not-cleared" >&2
  exit 1
fi
echo "sandbox-wipe.sh: nothing ${SANDBOX_USER}-owned left in ${#DIRS[@]} directories, SysV IPC or kernel keyrings"
