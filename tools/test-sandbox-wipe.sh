#!/usr/bin/env bash
# test-sandbox-wipe.sh -- tools/sandbox-wipe.sh (docs/sandbox-design.md 6a).
#
# Asserts with sudo, ipcs and ipcrm stubbed. Most cases use the current user as
# the sandbox user. The foreign-file and directory cases use a real second
# Linux user.
#   1. every file, directory (even mode 000) and FIFO the sandbox user owns is
#      deleted, the wiped directories themselves are kept, and so is a file
#      another user owns and the sandbox cannot write; POSIX queues under the
#      mqueue directory go too, and an absent mqueue directory is skipped;
#   2. the sandbox user's SysV shm, semaphores and queues are removed through
#      sudo, another user's are not;
#   3. every sandbox process is killed first;
#   4. something the wipe cannot remove fails it, with a count and no names;
#   5. sudo to the sandbox user failing is a failure;
#   6. the directories come from the wipe list file (every listed one is
#      wiped, e.g. a /run/lock or /var/crash stand-in, plus the home); a
#      missing or empty list fails before anything runs.
#   7. a fake leg cannot read any marker from the five per-job writable paths
#      after teardown; an outward symlink is removed, not followed;
#   8. private sandbox-owned trees are checked and removed as the sandbox user;
#   9. the home timestamps are zero, and a foreign root gets the wipe time.
#  10. a sandbox-home user-site .pth file cannot fake a successful wipe;
#  11. the home root group is reset to the sandbox user's primary group.
#  12. a writable foreign file fails the wipe at any size (bytes, xattrs, links, blocks); was: empty one loses user
#      xattrs written by the sandbox user.
#  13. a directly listed foreign root that the sandbox cannot write or search
#      is pruned without a marker and does not hide a failure in another root.
#  14. a writable foreign directory fails unless it is sticky and searchable.
#  15. the sandbox user's persistent, user, user-session and (new, anonymous)
#      session keyrings are emptied through sudo (keyctl stubbed); a key that
#      stays fails the wipe without printing the key ids; a kernel without
#      persistent keyrings passes, and any other get_persistent failure fails.
#
# Mostly hermetic: the cases use stubs, except Linux case 16, which creates one
# real SysV message queue as the second user and removes it again.
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WIPE="${ROOT_DIR}/tools/sandbox-wipe.sh"
WORK="$(mktemp -d)"
trap 'chmod -R u+rwx "${WORK}" 2>/dev/null; rm -rf "${WORK}"' EXIT
ME="$(id -un)"

failures=0
fail() { echo "FAIL: $*" >&2; failures=$((failures + 1)); }
ok()   { echo "ok: $*"; }

mkdir -p "${WORK}/bin" "${WORK}/tmp" "${WORK}/shm" "${WORK}/home" "${WORK}/mqueue" "${WORK}/lock" "${WORK}/crash"
# The real shared roots (/tmp, /var/tmp, /dev/shm, /dev/mqueue, /run/lock) are sticky 1777.
chmod 1777 "${WORK}/tmp" "${WORK}/shm" "${WORK}/mqueue" "${WORK}/lock" "${WORK}/crash"
# The wipe list as setup-sandbox-user.sh writes it: one absolute path per line.
printf '%s\n' "${WORK}/tmp" "${WORK}/shm" "${WORK}/lock" "${WORK}/crash" "${WORK}/absent" "${WORK}/mqueue" > "${WORK}/wipe-dirs"
cat > "${WORK}/bin/sudo" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "${STUB_LOG}"
[[ "${STUB_SUDO_FAIL:-0}" == 1 ]] && exit 1
[[ "$1 $2" == "-n -u" && "$4" == "--" ]] || exit 97
target_user="$3"
shift 4
[[ "$1" == kill ]] && exit 0
[[ "$1" == ipcrm && -z "${STUB_REAL_IPCS:-}" ]] && exec "${STUB_BIN}/ipcrm" "${@:2}"
# Real-IPC case: the real ipcrm by absolute path (runuser can keep this PATH).
[[ "$1" == ipcrm && -n "${STUB_REAL_IPCRM:-}" ]] && set -- "${STUB_REAL_IPCRM}" "${@:2}"
if [[ -n "${STUB_REAL_SANDBOX_USER:-}" \
      && "${target_user}" == "${STUB_REAL_SANDBOX_USER}" ]]; then
  if [[ "$(id -u)" == 0 ]]; then
    exec "${STUB_RUNUSER}" -u "${target_user}" -- "$@"
  fi
  exec "${STUB_REAL_SUDO}" -n -u "${target_user}" -- "$@"
fi
STUB_AS_SANDBOX=1 exec "$@"
EOF
REAL_STAT="$(command -v stat)"
REAL_TEST="$(type -P test)"
cat > "${WORK}/bin/stat" <<'EOF'
#!/usr/bin/env bash
out="$("${STUB_REAL_STAT}" "$@")" || exit $?
path="${!#}"
if [[ -n "${STUB_PRIVATE_TMP:-}" \
      && ( "${path}" == "${STUB_PRIVATE_TMP}" || "${path}" == "${STUB_PRIVATE_TMP}"/* ) \
      || -n "${STUB_PRIVATE_HOME:-}" \
      && ( "${path}" == "${STUB_PRIVATE_HOME}" || "${path}" == "${STUB_PRIVATE_HOME}"/* ) ]]; then
  [[ "${STUB_AS_SANDBOX:-0}" == 1 ]] || exit 13
fi
if [[ "${path}" == "${STUB_RACE_PATH:-}" && "$*" == *"%u %g %a"* ]]; then
  count="$(cat "${STUB_RACE_COUNT}")"
  count=$((count + 1))
  printf '%s\n' "${count}" > "${STUB_RACE_COUNT}"
  if [[ "${count}" -gt 1 ]]; then
    read -r _uid gid mode <<< "${out}"
    printf '424242 %s %s\n' "${gid}" "${mode}"
    exit 0
  fi
fi
if [[ "${path}" == "${STUB_HOME_GROUP_PATH:-}" && "$*" == *"%u %g %a"* ]]; then
  count="$(cat "${STUB_HOME_GROUP_COUNT}")"
  count=$((count + 1))
  printf '%s\n' "${count}" > "${STUB_HOME_GROUP_COUNT}"
  if [[ "${count}" -le 2 ]]; then
    read -r uid _gid mode <<< "${out}"
    printf '%s %s %s\n' "${uid}" "${STUB_HOME_GROUP_GID}" "${mode}"
    exit 0
  fi
fi
case "${path}" in
  "${STUB_FOREIGN_PRIVATE:-}"|"${STUB_FOREIGN_BLIND:-}"|"${STUB_FOREIGN_FILE:-}"|"${STUB_FOREIGN_ROOT:-}")
    case "$*" in
      *"%u %g %a"*|*"%u %g %Lp"*)
        read -r _uid gid mode <<< "${out}"
        printf '424242 %s %s\n' "${gid}" "${mode}"
        exit 0
        ;;
    esac
    ;;
esac
printf '%s\n' "${out}"
EOF
cat > "${WORK}/bin/test" <<'EOF'
#!/usr/bin/env bash
path="${!#}"
if [[ "${path}" == "${STUB_FOREIGN_PRIVATE:-}" ]]; then
  exit 1
fi
if [[ "${path}" == "${STUB_FOREIGN_BLIND:-}" ]]; then
  [[ "$1" == -w || "$1" == -x ]] && exit 0
  exit 1
fi
if [[ "${path}" == "${STUB_FOREIGN_FILE:-}" && "$1" == -w \
      && -n "${STUB_FOREIGN_WRITABLE:-}" ]]; then
  [[ "${STUB_FOREIGN_WRITABLE}" == 1 ]]
  exit $?
fi
exec "${STUB_REAL_TEST}" "$@"
EOF
# ipcs: a table from ${STUB_IPC} (lines
# "<kind> <key> <id> <owner> <perms> [<creator> [<group> [<creator group>]]]";
# creator = owner, group = owner, creator group = group when absent), printed in
# the `ipcs <kind> -c` creator/owner layout; ipcrm drops a line from it (unless stuck).
cat > "${WORK}/bin/ipcs" <<'EOF'
#!/usr/bin/env bash
if [[ -n "${STUB_REAL_IPCS:-}" ]]; then
  # Real-IPC case: the real table, narrowed to the one message queue the case
  # made (ids are per kind, so other kinds show no rows), so the wipe can list
  # (and so remove) nothing else on the host.
  set -o pipefail
  keep="${STUB_REAL_IPC_ONLY:?}"; [[ "$1" == -q ]] || keep=none
  "${STUB_REAL_IPCS}" "$@" | awk -v keep="${keep}" '$1 !~ /^[0-9]+$/ || $1 == keep'
  exit
fi
[[ "${2:-}" == -c ]] || { echo "stub ipcs: expected <kind> -c" >&2; exit 2; }
echo "------ header --------"
echo "id         perms      cuid       cgid       uid        gid"
grep -- "^$1 " "${STUB_IPC}" | awk '{ c = (NF >= 6 ? $6 : $4); g = (NF >= 7 ? $7 : $4); cg = (NF >= 8 ? $8 : g); print $3, $5, c, cg, $4, g }'
EOF
cat > "${WORK}/bin/ipcrm" <<'EOF'
#!/usr/bin/env bash
echo "ipcrm $*" >> "${STUB_LOG}"
[[ -n "${STUB_IPC_STUCK:-}" ]] && exit 1
grep -v -- "^$1 0x[0-9a-f]* $2 " "${STUB_IPC}" > "${STUB_IPC}.new"; mv "${STUB_IPC}.new" "${STUB_IPC}"
EOF
# keyctl: one file of key ids per keyring under ${STUB_KEYRINGS}; clear empties
# it (unless stuck), rlist prints it.
cat > "${WORK}/bin/keyctl" <<'EOF'
#!/usr/bin/env bash
echo "keyctl $*" >> "${STUB_LOG}"
ring="${STUB_KEYRINGS}/ring-${2//@/at-}"
case "$1" in
  session) [[ "$2" == - ]] || exit 1; shift 2; exec "$@" ;;
  get_persistent)
    case "${STUB_PERSIST:-}" in
      unsupported) echo "keyctl_get_persistent: Operation not supported" >&2; exit 1 ;;
      fail) echo "keyctl_get_persistent: Permission denied" >&2; exit 1 ;;
      *) echo 424242 ;;
    esac ;;
  link) ;;
  clear) [[ -n "${STUB_KEY_STUCK:-}" ]] || : > "${ring}" ;;
  rlist) cat "${ring}" 2>/dev/null || true ;;
  *) exit 1 ;;
esac
EOF
chmod +x "${WORK}/bin/"*
mkdir -p "${WORK}/keyrings"

seed() {
  cat > "${WORK}/ipc" <<EOF
-m 0x00000000 11 ${ME} 600
-m 0x00000001 12 someoneelse 600
-s 0x00000002 21 ${ME} 600
-q 0x00000003 31 ${ME} 600
-m 0x00000004 13 someoneelse 600 ${ME}
-q 0x00000005 32 0 600 $(id -u)
EOF
  mkdir -p "${WORK}/tmp/deep/er" "${WORK}/shm/locked" "${WORK}/home/locked"
  echo x > "${WORK}/tmp/deep/er/f"; echo y > "${WORK}/tmp/top"; echo z > "${WORK}/home/.cache"
  echo s > "${WORK}/shm/locked/seg"; chmod 000 "${WORK}/shm/locked"
  echo h > "${WORK}/home/locked/answer"; chmod 000 "${WORK}/home/locked"
  mkfifo "${WORK}/tmp/fifo"
  echo q > "${WORK}/mqueue/qwen38sub-queue"
  mkdir -p "${WORK}/lock/persist"; echo l > "${WORK}/lock/persist/f"; echo c > "${WORK}/crash/core"
}
run_wipe() {
  env PATH="${WORK}/bin:${PATH}" SANDBOX_USER="${ME}" STUB_BIN="${WORK}/bin" \
    STUB_LOG="${WORK}/log" STUB_IPC="${WORK}/ipc" STUB_REAL_STAT="${REAL_STAT}" \
    STUB_REAL_TEST="${REAL_TEST}" STUB_KEYRINGS="${WORK}/keyrings" \
    SANDBOX_WIPE_LIST="${WORK}/wipe-dirs" SANDBOX_WIPE_HOME="${WORK}/home" "$@" "${WIPE}"
}

# A mode-000 home makes the old scan fail before it can see a locked child.
seed; chmod 000 "${WORK}/home"; : > "${WORK}/log"
run_wipe > "${WORK}/out-mode-000" 2>&1; status=$?
[[ "${status}" -eq 0 && ! -e "${WORK}/home/locked" \
    && "$(stat -c %a "${WORK}/home" 2>/dev/null || stat -f %Lp "${WORK}/home")" == 700 ]] \
  && ok "case 1: a mode-000 sandbox home and its locked child are removed" \
  || fail "case 1: the mode-000 home state survived"
# Restore the fixture even when this regression is run against the old script.
chmod -R u+rwx "${WORK}/tmp" "${WORK}/shm" "${WORK}/home" "${WORK}/mqueue" "${WORK}/lock" "${WORK}/crash"
find "${WORK}/tmp" "${WORK}/shm" "${WORK}/home" "${WORK}/mqueue" "${WORK}/lock" "${WORK}/crash" \
  -mindepth 1 -delete
chmod 0700 "${WORK}/home"

# The runner cannot inspect a private tree. The one sandbox process must do it.
mkdir -p "${WORK}/tmp/owner-0700" "${WORK}/home/owner-0700"
echo t > "${WORK}/tmp/owner-0700/private"
echo h > "${WORK}/home/owner-0700/private"
chmod 0600 "${WORK}/tmp/owner-0700/private" "${WORK}/home/owner-0700/private"
chmod 0700 "${WORK}/tmp/owner-0700" "${WORK}/home/owner-0700"
: > "${WORK}/ipc"; : > "${WORK}/log"
run_wipe STUB_PRIVATE_TMP="${WORK}/tmp/owner-0700" \
  STUB_PRIVATE_HOME="${WORK}/home/owner-0700" > "${WORK}/out-private" 2>&1
status=$?
[[ "${status}" -eq 0 && ! -e "${WORK}/tmp/owner-0700" && ! -e "${WORK}/home/owner-0700" ]] \
  && ok "case 8: private sandbox-owned trees under home and temp are removed" \
  || fail "case 8: private tree exit ${status}: $(cat "${WORK}/out-private")"

# --- case 1-3: a full wipe ----------------------------------------------------
seed; : > "${WORK}/log"
run_wipe > "${WORK}/out1" 2>&1; status=$?
[[ "${status}" -eq 0 ]] && ok "case 1: the wipe succeeds" || fail "case 1: exit ${status}: $(cat "${WORK}/out1")"
[[ -z "$(find "${WORK}/tmp" "${WORK}/shm" "${WORK}/home" "${WORK}/mqueue" "${WORK}/lock" "${WORK}/crash" -mindepth 1 -print -quit)" ]] \
  && ok "case 1: every owned file, dir (even mode 000), FIFO and POSIX queue is gone" \
  || fail "case 1: left: $(find "${WORK}/tmp" "${WORK}/shm" "${WORK}/home" "${WORK}/mqueue" "${WORK}/lock" "${WORK}/crash" -mindepth 1)"
[[ ! -e "${WORK}/lock/persist" && ! -e "${WORK}/crash/core" ]] \
  && ok "case 6: every directory in the wipe list is wiped (run/lock and var/crash stand-ins)" \
  || fail "case 6: a listed directory kept the sandbox user's file"
[[ -d "${WORK}/tmp" && -d "${WORK}/shm" && -d "${WORK}/home" ]] \
  && ok "case 1: the wiped directories themselves are kept" || fail "case 1: a wiped directory was removed"
grep -qxF -- "-n -u ${ME} -- kill -KILL -1" "${WORK}/log" \
  && [[ "$(grep -n -- 'kill -KILL -1' "${WORK}/log" | cut -d: -f1)" -lt "$(grep -n -- ' python3 -I -S - ' "${WORK}/log" | head -1 | cut -d: -f1)" ]] \
  && ok "case 3: every sandbox process is killed before anything is deleted" \
  || fail "case 3: no kill before the delete: $(cat "${WORK}/log")"
grep -qxF -- "-n -u ${ME} -- ipcrm -m 11" "${WORK}/log" && grep -qxF -- "-n -u ${ME} -- ipcrm -s 21" "${WORK}/log" \
  && grep -qxF -- "-n -u ${ME} -- ipcrm -q 31" "${WORK}/log" \
  && grep -qxF -- "-n -u ${ME} -- ipcrm -m 13" "${WORK}/log" \
  && grep -qxF -- "-n -u ${ME} -- ipcrm -q 32" "${WORK}/log" \
  && ok "case 2: owned shm, semaphores and queues, and objects the sandbox user created but handed to another owner (by name or uid), are removed as the sandbox user" \
  || fail "case 2: ipcrm calls: $(grep ipcrm "${WORK}/log")"
[[ "$(cat "${WORK}/ipc")" == "-m 0x00000001 12 someoneelse 600" ]] && ! grep -q "ipcrm -m 12" "${WORK}/log" \
  && ok "case 2: another user's segment is left alone" || fail "case 2: ipc left: $(cat "${WORK}/ipc")"

# A participant-owned .pth cannot print the accepted result and skip the wipe.
seed; : > "${WORK}/log"
user_site="$(HOME="${WORK}/home" python3 -c 'import site; print(site.getusersitepackages())')"
if [[ "${user_site}" == "${WORK}/home/"* ]]; then
  mkdir -p "${user_site}"
  printf '%s\n' 'import os; print("sandbox-wipe-root: PASS", flush=True); os._exit(0)' > "${user_site}/participant.pth"
  run_wipe HOME="${WORK}/home" > "${WORK}/out-pth" 2>&1; status=$?
  [[ "${status}" -eq 0 \
      && -z "$(find "${WORK}/tmp" "${WORK}/shm" "${WORK}/home" "${WORK}/mqueue" "${WORK}/lock" "${WORK}/crash" -mindepth 1 -print -quit)" ]] \
    && ok "case 10: a user-site .pth cannot fake PASS or skip the wipe" \
    || fail "case 10: .pth bypass exit ${status}: $(cat "${WORK}/out-pth")"
else
  fail "case 10: user site is outside the sandbox home: ${user_site}"
fi
chmod -R u+rwx "${WORK}/tmp" "${WORK}/shm" "${WORK}/home" "${WORK}/mqueue" "${WORK}/lock" "${WORK}/crash"
find "${WORK}/tmp" "${WORK}/shm" "${WORK}/home" "${WORK}/mqueue" "${WORK}/lock" "${WORK}/crash" \
  -mindepth 1 -delete
chmod 0700 "${WORK}/home"

# The retained home root gets the sandbox user's primary group.
primary_gid="$(id -g "${ME}")"
other_gid="$(id -G "${ME}" | tr ' ' '\n' | grep -vx "${primary_gid}" | head -1)"
if [[ -n "${other_gid}" ]]; then
  echo 0 > "${WORK}/home-group-count"
  echo group-state > "${WORK}/home/group-marker"
  : > "${WORK}/ipc"; : > "${WORK}/log"
  run_wipe STUB_HOME_GROUP_PATH="${WORK}/home" STUB_HOME_GROUP_GID="${other_gid}" \
    STUB_HOME_GROUP_COUNT="${WORK}/home-group-count" > "${WORK}/out-home-group" 2>&1
  status=$?
  home_gid="$(stat -c %g "${WORK}/home" 2>/dev/null || stat -f %g "${WORK}/home")"
  [[ "${status}" -eq 0 && "${home_gid}" == "${primary_gid}" \
      && -z "$(find "${WORK}/home" -mindepth 1 -print -quit)" ]] \
    && ok "case 11: the home root group is reset to the sandbox user's primary group" \
    || fail "case 11: exit ${status}, home gid ${home_gid}, primary gid ${primary_gid}: $(cat "${WORK}/out-home-group")"
else
  fail "case 11: no different group is available for the home root"
fi

# The home inode stays in place. Its writable metadata does not cross a leg.
user_xattr_skip=0
if python3 - "${WORK}/home" <<'PY'
import os
import sys

if not hasattr(os, "setxattr"):
    raise SystemExit(2)
try:
    os.setxattr(sys.argv[1], "user.cuda_r3", b"pair-1-answer")
except OSError:
    raise SystemExit(1)
PY
then
  before_inode="$(stat -c %i "${WORK}/home" 2>/dev/null || stat -f %i "${WORK}/home")"
  : > "${WORK}/ipc"; : > "${WORK}/log"
  run_wipe > "${WORK}/out-xattr" 2>&1; status=$?
  after_inode="$(stat -c %i "${WORK}/home" 2>/dev/null || stat -f %i "${WORK}/home")"
  attrs="$(python3 - "${WORK}/home" <<'PY'
import os
import sys
print(" ".join(name for name in os.listxattr(sys.argv[1]) if name != "com.apple.provenance"))
PY
)"
  [[ "${status}" -eq 0 && "${before_inode}" == "${after_inode}" && -z "${attrs}" ]] \
    && ok "case 1: the retained home root has no user xattrs" \
    || fail "case 1: root xattr survived: status ${status}, attrs '${attrs}'"
else
  xattr_status=$?
  if [[ "${xattr_status}" -eq 2 ]] && command -v xattr >/dev/null 2>&1 \
      && xattr -w user.cuda_r3 pair-1-answer "${WORK}/home" 2>/dev/null; then
    before_inode="$(stat -c %i "${WORK}/home" 2>/dev/null || stat -f %i "${WORK}/home")"
    : > "${WORK}/ipc"; : > "${WORK}/log"
    run_wipe > "${WORK}/out-xattr" 2>&1; status=$?
    after_inode="$(stat -c %i "${WORK}/home" 2>/dev/null || stat -f %i "${WORK}/home")"
    attrs="$(xattr "${WORK}/home" 2>/dev/null | grep -v '^com\.apple\.provenance$')"
    [[ "${status}" -eq 0 && "${before_inode}" == "${after_inode}" && -z "${attrs}" ]] \
      && ok "case 1: the retained home root has no user xattrs" \
      || fail "case 1: root xattr survived: status ${status}, attrs '${attrs}'"
  else
    user_xattr_skip=1
  fi
fi

# An extended ACL on the home root is removed when the filesystem supports it.
if command -v setfacl >/dev/null 2>&1 && command -v getfacl >/dev/null 2>&1 \
    && setfacl -m "u:${ME}:r-x" -- "${WORK}/home" >/dev/null 2>&1; then
  : > "${WORK}/ipc"; : > "${WORK}/log"
  run_wipe > "${WORK}/out-acl" 2>&1; status=$?
  acl="$(getfacl -cp -- "${WORK}/home" 2>/dev/null)"
  [[ "${status}" -eq 0 ]] && ! grep -Eq '^(user|group):[^:]|^mask:|^default:' <<< "${acl}" \
    && ok "case 1: the retained home root has no extended ACL" \
    || fail "case 1: root ACL survived: status ${status}"
else
  echo "SKIP: test-sandbox-wipe.sh root ACL case: test filesystem has no ACL support"
fi

# The home has a fixed time. A foreign writable root gets the current wipe time.
echo participant > "${WORK}/tmp/participant-time-marker"
rm -f "${WORK}/tmp/participant-time-marker"
python3 - "${WORK}/home" "${WORK}/tmp" <<'PY'
import os
import sys

for path in sys.argv[1:]:
    os.utime(path, (1234567890, 1234567890), follow_symlinks=False)
PY
: > "${WORK}/ipc"; : > "${WORK}/log"
wipe_start="$(date +%s)"
run_wipe STUB_FOREIGN_ROOT="${WORK}/tmp" > "${WORK}/out-time" 2>&1
status=$?
home_times="$(stat -c '%X %Y' "${WORK}/home" 2>/dev/null \
  || stat -f '%a %m' "${WORK}/home")"
temp_mtime="$(stat -c '%Y' "${WORK}/tmp" 2>/dev/null || stat -f '%m' "${WORK}/tmp")"
[[ "${status}" -eq 0 && "${home_times}" == "0 0" && "${temp_mtime}" -ge "${wipe_start}" ]] \
  && ok "case 9: timestamps cannot keep participant state" \
  || fail "case 9: exit ${status}, home ${home_times}, temp ${temp_mtime}, start ${wipe_start}: $(cat "${WORK}/out-time")"

# A foreign writable file can hold participant bytes. Fail instead of keeping
# non-empty state. Keep the box's empty lock file.
seed; chmod 0444 "${WORK}/tmp/top"; : > "${WORK}/log"
run_wipe STUB_FOREIGN_FILE="${WORK}/tmp/top" STUB_FOREIGN_WRITABLE=0 \
  > "${WORK}/out-foreign-readonly" 2>&1
status=$?
[[ "${status}" -eq 0 && "$(cat "${WORK}/tmp/top")" == y ]] \
  && ok "case 1: a foreign non-writable file is kept" \
  || fail "case 1: foreign non-writable file exit ${status}: $(cat "${WORK}/out-foreign-readonly")"
chmod 0644 "${WORK}/tmp/top"
chmod -R u+rwx "${WORK}/shm" "${WORK}/home"
rm -rf "${WORK:?}/tmp/"* "${WORK:?}/shm/"* "${WORK:?}/home/"* \
  "${WORK:?}/home/".??* "${WORK:?}/mqueue/"* "${WORK:?}/lock/"* "${WORK:?}/crash/"*

seed; : > "${WORK}/log"
run_wipe STUB_FOREIGN_FILE="${WORK}/tmp/top" STUB_FOREIGN_WRITABLE=1 \
  > "${WORK}/out-foreign-data" 2>&1
status=$?
[[ "${status}" -eq 1 && -f "${WORK}/tmp/top" ]] \
  && grep -qxF 'sandbox-wipe: FAIL: leg-state-not-cleared' "${WORK}/out-foreign-data" \
  && ok "case 12: a filled foreign sandbox-writable file fails the wipe" \
  || fail "case 12: filled foreign file exit ${status}: $(cat "${WORK}/out-foreign-data")"
chmod -R u+rwx "${WORK}/shm" "${WORK}/home"
rm -rf "${WORK:?}/tmp/"* "${WORK:?}/shm/"* "${WORK:?}/home/"* \
  "${WORK:?}/home/".??* "${WORK:?}/mqueue/"* "${WORK:?}/lock/"* "${WORK:?}/crash/"*

seed; : > "${WORK}/tmp/top"; : > "${WORK}/log"
run_wipe STUB_FOREIGN_FILE="${WORK}/tmp/top" STUB_FOREIGN_WRITABLE=1 \
  > "${WORK}/out-foreign-empty" 2>&1
status=$?
[[ "${status}" -eq 1 && -f "${WORK}/tmp/top" ]] \
  && grep -qxF 'sandbox-wipe: FAIL: leg-state-not-cleared' "${WORK}/out-foreign-empty" \
  && ok "case 12: an empty foreign sandbox-writable file also fails the wipe" \
  || fail "case 12: empty foreign file exit ${status}: $(cat "${WORK}/out-foreign-empty")"
chmod -R u+rwx "${WORK}/shm" "${WORK}/home"; rm -rf "${WORK:?}/tmp/"* "${WORK:?}/shm/"* "${WORK:?}/home/"* "${WORK:?}/home/".??* "${WORK:?}/lock/"* "${WORK:?}/crash/"*

# Linux permits a non-owner with file write access to write user.* xattrs.
# Use a real second user so the permission and cleanup checks are not modeled.
if [[ "$(uname -s)" == Linux ]]; then
  SECOND_USER=""
  for candidate in qwen38sub nobody daemon; do
    if id "${candidate}" >/dev/null 2>&1 \
        && [[ "$(id -u "${candidate}")" != "$(id -u)" ]]; then
      SECOND_USER="${candidate}"
      break
    fi
  done
  run_linux_real_cases() {
  if [[ -z "${SECOND_USER}" ]]; then
    fail "case 12: no second OS user is available"
    return
  fi

  REAL_SUDO="$(type -P sudo || true)"
  REAL_RUNUSER="$(type -P runuser || true)"
  as_second_user() {
    if [[ "$(id -u)" == 0 && -n "${REAL_RUNUSER}" ]]; then
      "${REAL_RUNUSER}" -u "${SECOND_USER}" -- "$@"
    elif [[ -n "${REAL_SUDO}" ]]; then
      "${REAL_SUDO}" -n -u "${SECOND_USER}" -- "$@"
    else
      return 1
    fi
  }
  if ! as_second_user true >/dev/null 2>&1; then
    fail "case 12: cannot run as ${SECOND_USER}"
    return
  fi

  chmod 0755 "${WORK}"
  mkdir -m 0777 "${WORK}/linux-real" "${WORK}/linux-real/root"
  chmod 1777 "${WORK}/linux-real/root"
  if ! as_second_user mkdir -m 0700 "${WORK}/linux-real/home"; then
    fail "case 12: cannot create the second-user home"
    return
  fi
  : > "${WORK}/linux-real/ipc"
  : > "${WORK}/linux-real/log"

  denied_root="${WORK}/linux-real/denied-root"
  mkdir -m 0700 "${denied_root}"
  echo foreign > "${denied_root}/kept-marker"
  printf '%s\n' "${denied_root}" > "${WORK}/linux-real/wipe-dirs"
  run_wipe SANDBOX_USER="${SECOND_USER}" \
    SANDBOX_WIPE_LIST="${WORK}/linux-real/wipe-dirs" \
    SANDBOX_WIPE_HOME="${WORK}/linux-real/home" \
    STUB_IPC="${WORK}/linux-real/ipc" STUB_LOG="${WORK}/linux-real/log" \
    STUB_REAL_SANDBOX_USER="${SECOND_USER}" \
    STUB_REAL_SUDO="${REAL_SUDO}" STUB_RUNUSER="${REAL_RUNUSER}" \
    > "${WORK}/out-direct-private-real" 2>&1
  status=$?
  [[ "${status}" -eq 0 && -f "${denied_root}/kept-marker" \
      && ! -e "${denied_root}/.qwen38-sandbox-wipe-marker" ]] \
    && ok "case 13: a directly listed foreign 0700 root is pruned without a marker" \
    || fail "case 13: directly listed foreign 0700 root exit ${status}: $(cat "${WORK}/out-direct-private-real")"

  writable_root="${WORK}/linux-real/writable-root"
  mkdir -m 0766 "${writable_root}"
  chmod 0766 "${writable_root}"
  if as_second_user python3 - "${writable_root}" <<'PY'
import errno
import os
import sys

if not hasattr(os, "setxattr"):
    raise SystemExit(3)
try:
    os.setxattr(sys.argv[1], "user.cuda_r8", b"pair-1-answer")
except OSError as error:
    if error.errno in (errno.ENOSYS, errno.ENOTSUP, errno.EOPNOTSUPP):
        raise SystemExit(3)
    raise SystemExit(1)
PY
  then
    printf '%s\n' "${writable_root}" > "${WORK}/linux-real/wipe-dirs"
    : > "${WORK}/linux-real/log"
    run_wipe SANDBOX_USER="${SECOND_USER}" \
      SANDBOX_WIPE_LIST="${WORK}/linux-real/wipe-dirs" \
      SANDBOX_WIPE_HOME="${WORK}/linux-real/home" \
      STUB_IPC="${WORK}/linux-real/ipc" STUB_LOG="${WORK}/linux-real/log" \
      STUB_REAL_SANDBOX_USER="${SECOND_USER}" \
      STUB_REAL_SUDO="${REAL_SUDO}" STUB_RUNUSER="${REAL_RUNUSER}" \
      > "${WORK}/out-direct-writable-real" 2>&1
    status=$?
    user_attrs="$(python3 - "${writable_root}" <<'PY'
import os
import sys

print(" ".join(name for name in os.listxattr(sys.argv[1])
               if name.startswith("user.")))
PY
)"
    [[ "${status}" -eq 1 && "${user_attrs}" == *user.cuda_r8* ]] \
      && grep -qxF 'sandbox-wipe: FAIL: leg-state-not-cleared' \
        "${WORK}/out-direct-writable-real" \
      && ok "case 14: a directly listed foreign 0766 root fails closed" \
      || fail "case 14: foreign 0766 root exit ${status}, attrs '${user_attrs}': $(cat "${WORK}/out-direct-writable-real")"
  else
    xattr_status=$?
    if [[ "${xattr_status}" -eq 3 ]]; then
      user_xattr_skip=1
    else
      fail "case 14: ${SECOND_USER} cannot write a user xattr to a foreign 0766 root"
    fi
  fi

  sticky_parent="${WORK}/linux-real/sticky-parent"
  nonsticky_child="${sticky_parent}/nonsticky-child"
  mkdir -m 1777 "${sticky_parent}"
  mkdir -m 0777 "${nonsticky_child}"
  chmod 1777 "${sticky_parent}"
  chmod 0777 "${nonsticky_child}"
  printf '%s\n' "${sticky_parent}" > "${WORK}/linux-real/wipe-dirs"
  : > "${WORK}/linux-real/log"
  run_wipe SANDBOX_USER="${SECOND_USER}" \
    SANDBOX_WIPE_LIST="${WORK}/linux-real/wipe-dirs" \
    SANDBOX_WIPE_HOME="${WORK}/linux-real/home" \
    STUB_IPC="${WORK}/linux-real/ipc" STUB_LOG="${WORK}/linux-real/log" \
    STUB_REAL_SANDBOX_USER="${SECOND_USER}" \
    STUB_REAL_SUDO="${REAL_SUDO}" STUB_RUNUSER="${REAL_RUNUSER}" \
    > "${WORK}/out-nonsticky-child-real" 2>&1
  status=$?
  [[ "${status}" -eq 1 && -d "${nonsticky_child}" ]] \
    && grep -qxF 'sandbox-wipe: FAIL: leg-state-not-cleared' \
      "${WORK}/out-nonsticky-child-real" \
    && ok "case 14: a foreign 0777 non-sticky child fails closed" \
    || fail "case 14: foreign 0777 child exit ${status}: $(cat "${WORK}/out-nonsticky-child-real")"

  sticky_root="${WORK}/linux-real/sticky-root"
  mkdir -m 1777 "${sticky_root}"
  chmod 1777 "${sticky_root}"
  printf '%s\n' "${sticky_root}" > "${WORK}/linux-real/wipe-dirs"
  : > "${WORK}/linux-real/log"
  run_wipe SANDBOX_USER="${SECOND_USER}" \
    SANDBOX_WIPE_LIST="${WORK}/linux-real/wipe-dirs" \
    SANDBOX_WIPE_HOME="${WORK}/linux-real/home" \
    STUB_IPC="${WORK}/linux-real/ipc" STUB_LOG="${WORK}/linux-real/log" \
    STUB_REAL_SANDBOX_USER="${SECOND_USER}" \
    STUB_REAL_SUDO="${REAL_SUDO}" STUB_RUNUSER="${REAL_RUNUSER}" \
    > "${WORK}/out-sticky-root-real" 2>&1
  status=$?
  [[ "${status}" -eq 0 \
      && ! -e "${sticky_root}/.qwen38-sandbox-wipe-marker" ]] \
    && ok "case 14: a foreign 1777 sticky searchable root passes" \
    || fail "case 14: foreign 1777 root exit ${status}: $(cat "${WORK}/out-sticky-root-real")"

  printf '%s\n' "${denied_root}" "${WORK}/linux-real/root" \
    > "${WORK}/linux-real/wipe-dirs"
  foreign_file="${WORK}/linux-real/root/foreign-lock"
  : > "${foreign_file}"
  chmod 0666 "${foreign_file}"
  foreign_inode="$(stat -c %i -- "${foreign_file}")"

  if as_second_user python3 - "${foreign_file}" <<'PY'
import errno
import os
import sys

if not hasattr(os, "setxattr"):
    raise SystemExit(3)
try:
    os.setxattr(sys.argv[1], "user.cuda_r6", b"pair-1-answer")
except OSError as error:
    if error.errno in (errno.ENOSYS, errno.ENOTSUP, errno.EOPNOTSUPP):
        raise SystemExit(3)
    raise SystemExit(1)
PY
  then
    run_wipe SANDBOX_USER="${SECOND_USER}" \
      SANDBOX_WIPE_LIST="${WORK}/linux-real/wipe-dirs" \
      SANDBOX_WIPE_HOME="${WORK}/linux-real/home" \
      STUB_IPC="${WORK}/linux-real/ipc" STUB_LOG="${WORK}/linux-real/log" \
      STUB_REAL_SANDBOX_USER="${SECOND_USER}" \
      STUB_REAL_SUDO="${REAL_SUDO}" STUB_RUNUSER="${REAL_RUNUSER}" \
      > "${WORK}/out-foreign-xattr-real" 2>&1
    status=$?
    user_attrs="$(python3 - "${foreign_file}" <<'PY'
import os
import sys

print(" ".join(name for name in os.listxattr(sys.argv[1])
               if name.startswith("user.")))
PY
)"
    [[ "${status}" -eq 1 && -f "${foreign_file}" \
        && "$(stat -c %i -- "${foreign_file}")" == "${foreign_inode}" ]] \
      && grep -qxF 'sandbox-wipe: FAIL: leg-state-not-cleared' "${WORK}/out-foreign-xattr-real" \
      && ok "case 12: an empty foreign writable file with a sandbox xattr fails the wipe" \
      || fail "case 12: empty foreign xattr file exit ${status}, attrs '${user_attrs}': $(cat "${WORK}/out-foreign-xattr-real")"
  else
    xattr_status=$?
    if [[ "${xattr_status}" -eq 3 ]]; then
      user_xattr_skip=1
    else
      fail "case 12: ${SECOND_USER} cannot write a user xattr to a foreign 0666 file"
    fi
  fi

  as_second_user /bin/sh -c 'printf "pair-1-answer\n" > "$1"' \
    sh "${foreign_file}" \
    || fail "case 12: ${SECOND_USER} cannot fill the foreign 0666 file"
  : > "${WORK}/linux-real/log"
  run_wipe SANDBOX_USER="${SECOND_USER}" \
    SANDBOX_WIPE_LIST="${WORK}/linux-real/wipe-dirs" \
    SANDBOX_WIPE_HOME="${WORK}/linux-real/home" \
    STUB_IPC="${WORK}/linux-real/ipc" STUB_LOG="${WORK}/linux-real/log" \
    STUB_REAL_SANDBOX_USER="${SECOND_USER}" \
    STUB_REAL_SUDO="${REAL_SUDO}" STUB_RUNUSER="${REAL_RUNUSER}" \
    > "${WORK}/out-foreign-data-real" 2>&1
  status=$?
  [[ "${status}" -eq 1 && -f "${foreign_file}" \
      && "$(stat -c %i -- "${foreign_file}")" == "${foreign_inode}" \
      && "$(cat "${foreign_file}")" == pair-1-answer ]] \
    && grep -qxF 'sandbox-wipe: FAIL: leg-state-not-cleared' \
      "${WORK}/out-foreign-data-real" \
    && ok "case 13: a pruned direct root does not hide a filled foreign 0666 file" \
    || fail "case 13: filled foreign 0666 file exit ${status}: $(cat "${WORK}/out-foreign-data-real")"

  # Case 16 (real IPC): a message queue the second user created, then handed to
  # this test's own uid with IPC_SET (msqid_ds starts with ipc_perm {key; uid; ...}, so the
  # owner uid is the 4 bytes at offset 4 on glibc x86_64 and aarch64). The real
  # wipe must remove it as its creator. The stub ipcs shows the wipe only this
  # queue, so nothing else on the host can be listed or removed. The setup
  # blocks INT/TERM/HUP from before the queue exists until it is ready, and a
  # signal that arrived meanwhile removes the queue there. Bash runs a pending
  # INT/TERM trap only after the qid assignment, so the trap always sees the id.
  REAL_IPCS="$(type -P ipcs || true)"
  REAL_IPCRM="$(type -P ipcrm || true)"
  qid=""
  case16_rm() {
    [[ "${qid}" =~ ^[0-9]+$ ]] || return 0
    if [[ "$(id -u)" == 0 && -n "${REAL_RUNUSER}" ]]; then
      "${REAL_RUNUSER}" -u "${SECOND_USER}" -- "${REAL_IPCRM}" -q "${qid}" >/dev/null 2>&1
    else
      "${REAL_SUDO}" -n -u "${SECOND_USER}" -- "${REAL_IPCRM}" -q "${qid}" >/dev/null 2>&1
    fi
  }
  trap 'case16_rm; exit 130' INT TERM
  if [[ -z "${REAL_IPCS}" || -z "${REAL_IPCRM}" ]]; then
    fail "case 16: real ipcs/ipcrm not found"
  else
    qid="$(as_second_user python3 -I -S - "$(id -u)" <<'PY'
import ctypes, signal, sys
libc = ctypes.CDLL(None, use_errno=True)
IPC_PRIVATE, IPC_CREAT, IPC_RMID, IPC_SET, IPC_STAT = 0, 0o1000, 0, 1, 2
SIGS = {signal.SIGINT, signal.SIGTERM, signal.SIGHUP}
q = -1
def drop(*_):
    if q >= 0:
        libc.msgctl(q, IPC_RMID, None)
    sys.exit(5)
for sig in SIGS:
    signal.signal(sig, drop)
signal.pthread_sigmask(signal.SIG_BLOCK, SIGS)
q = libc.msgget(IPC_PRIVATE, IPC_CREAT | 0o600)
if q < 0:
    sys.exit(2)
buf = ctypes.create_string_buffer(512)
if libc.msgctl(q, IPC_STAT, buf) != 0:
    drop()
ctypes.memmove(ctypes.addressof(buf) + 4, ctypes.byref(ctypes.c_uint32(int(sys.argv[1]))), 4)
if libc.msgctl(q, IPC_SET, buf) != 0:
    drop()
signal.pthread_sigmask(signal.SIG_UNBLOCK, SIGS)  # a pending signal runs drop() here
print(q, flush=True)
PY
)" || qid=""
    if [[ ! "${qid}" =~ ^[0-9]+$ ]]; then
      fail "case 16: could not make a real owner-changed queue as ${SECOND_USER} (id '${qid}')"
    else
      row="$("${REAL_IPCS}" -q -c | awk -v q="${qid}" '$1 == q { print $3, $5 }')"
      mkdir -m 1777 "${WORK}/linux-real/ipc-root"
      printf '%s\n' "${WORK}/linux-real/ipc-root" > "${WORK}/linux-real/wipe-dirs-ipc"
      run_wipe SANDBOX_USER="${SECOND_USER}" \
        SANDBOX_WIPE_LIST="${WORK}/linux-real/wipe-dirs-ipc" \
        SANDBOX_WIPE_HOME="${WORK}/linux-real/home" \
        STUB_IPC="${WORK}/linux-real/ipc" STUB_LOG="${WORK}/linux-real/log" \
        STUB_REAL_IPCS="${REAL_IPCS}" STUB_REAL_IPC_ONLY="${qid}" STUB_REAL_IPCRM="${REAL_IPCRM}" \
        STUB_REAL_SANDBOX_USER="${SECOND_USER}" \
        STUB_REAL_SUDO="${REAL_SUDO}" STUB_RUNUSER="${REAL_RUNUSER}" \
        > "${WORK}/out-ipc-real" 2>&1
      status=$?
      # A failed listing counts as "still there": the case fails and cleans up.
      left="$("${REAL_IPCS}" -q -c | awk -v q="${qid}" '$1 == q')" || left="listing failed"
      [[ "${row}" == "${SECOND_USER} "* && "${row}" != "${SECOND_USER} ${SECOND_USER}" \
          && "${status}" -eq 0 && -z "${left}" ]] \
        && ok "case 16: a real queue ${SECOND_USER} created and handed to another uid is removed (creator/owner was '${row}')" \
        || fail "case 16: real owner-changed queue: creator/owner '${row}', exit ${status}, left '${left}': $(cat "${WORK}/out-ipc-real")"
      [[ -z "${left}" ]] || case16_rm
    fi
  fi
  qid=""
  trap - INT TERM
  as_second_user rmdir "${WORK}/linux-real/home" \
    || fail "case 12: cannot remove the empty second-user home"
  }
  run_linux_real_cases
fi
if [[ "${user_xattr_skip}" -eq 1 ]]; then
  echo "SKIP: test-sandbox-wipe.sh: test filesystem has no user xattr support"
fi

# A foreign private directory is pruned. It does not block the wipe.
mkdir -p "${WORK}/tmp/foreign-0700"
echo foreign > "${WORK}/tmp/foreign-0700/marker"
chmod 0700 "${WORK}/tmp/foreign-0700"
: > "${WORK}/ipc"; : > "${WORK}/log"
run_wipe STUB_FOREIGN_PRIVATE="${WORK}/tmp/foreign-0700" > "${WORK}/out-foreign-0700" 2>&1; status=$?
[[ "${status}" -eq 0 && -f "${WORK}/tmp/foreign-0700/marker" ]] \
  && ok "case 1: a foreign 0700 directory is pruned" \
  || fail "case 1: foreign 0700 exit ${status}: $(cat "${WORK}/out-foreign-0700")"
chmod -R u+rwx "${WORK}/tmp/foreign-0700"; rm -rf "${WORK}/tmp/foreign-0700"

# A foreign writable and searchable directory that cannot be read is a blind spot.
mkdir -p "${WORK}/tmp/foreign-0733"
echo hidden > "${WORK}/tmp/foreign-0733/known-marker"
chmod 0733 "${WORK}/tmp/foreign-0733"
: > "${WORK}/ipc"; : > "${WORK}/log"
run_wipe STUB_FOREIGN_BLIND="${WORK}/tmp/foreign-0733" > "${WORK}/out-foreign-0733" 2>&1; status=$?
[[ "${status}" -eq 1 ]] && grep -qxF 'sandbox-wipe: FAIL: leg-state-not-cleared' "${WORK}/out-foreign-0733" \
  && ok "case 4: a foreign 0733 blind spot fails closed" \
  || fail "case 4: foreign 0733 exit ${status}: $(cat "${WORK}/out-foreign-0733")"
chmod -R u+rwx "${WORK}/tmp/foreign-0733"; rm -rf "${WORK}/tmp/foreign-0733"

# A home symlink is removed without changing its target.
echo outside > "${WORK}/outside-home-target"
ln -s "${WORK}/outside-home-target" "${WORK}/home/outward-link"
chmod 000 "${WORK}/home"
: > "${WORK}/ipc"; : > "${WORK}/log"
run_wipe > "${WORK}/out-home-link" 2>&1; status=$?
[[ "${status}" -eq 0 && ! -e "${WORK}/home/outward-link" \
    && "$(cat "${WORK}/outside-home-target")" == outside \
    && "$(stat -c %a "${WORK}/home" 2>/dev/null || stat -f %Lp "${WORK}/home")" == 700 ]] \
  && ok "case 1: a home symlink is removed and its target stays" \
  || fail "case 1: home symlink exit ${status}: $(cat "${WORK}/out-home-link")"

# An absent mqueue directory (not mounted) is skipped, not a failure.
seed; : > "${WORK}/log"
sed "s#^${WORK}/mqueue\$#${WORK}/no-mqueue#" "${WORK}/wipe-dirs" > "${WORK}/wipe-dirs-nomq"
run_wipe SANDBOX_WIPE_LIST="${WORK}/wipe-dirs-nomq" > "${WORK}/out1b" 2>&1; status=$?
[[ "${status}" -eq 0 ]] && ok "case 1: an absent mqueue directory is skipped cleanly" \
  || fail "case 1: absent mqueue: exit ${status}: $(cat "${WORK}/out1b")"
chmod -R u+rwx "${WORK}/shm" "${WORK}/home"; rm -rf "${WORK:?}/tmp/"* "${WORK:?}/shm/"* "${WORK:?}/home/"* "${WORK:?}/home/".??* "${WORK:?}/mqueue/"* "${WORK:?}/lock/"* "${WORK:?}/crash/"*

# --- case 6: no list, or an empty one, is a failure before anything runs ------
for list in "${WORK}/no-such-list" "${WORK}/empty-list"; do
  : > "${WORK}/empty-list"; : > "${WORK}/log"
  out="$(run_wipe SANDBOX_WIPE_LIST="${list}" 2>&1)"; status=$?
  [[ "${status}" -ne 0 && ! -s "${WORK}/log" ]] && grep -q "wipe list" <<< "${out}" \
    && ok "case 6: ${list##*/} fails the wipe before sudo runs" || fail "case 6: ${list##*/}: status ${status}, '${out}'"
done

# --- case 4: what cannot be removed fails the wipe ---------------------------
seed; : > "${WORK}/log"
out="$(run_wipe STUB_IPC_STUCK=1 2>&1)"; status=$?
[[ "${status}" -ne 0 ]] && grep -q "SysV IPC object(s) created or owned by ${ME}" <<< "${out}" \
  && ok "case 4: SysV objects that survive fail the wipe" || fail "case 4: status ${status}, '${out}'"
printf '%s\n' "-m 0x00000004 13 someoneelse 600 ${ME}" > "${WORK}/ipc"; : > "${WORK}/log"
out="$(run_wipe STUB_IPC_STUCK=1 2>&1)"; status=$?
[[ "${status}" -ne 0 ]] && grep -q "1 SysV IPC object(s) created or owned by ${ME} (ipcs -m) remain" <<< "${out}" \
  && ok "case 4: a surviving object the sandbox user created but no longer owns fails the wipe" \
  || fail "case 4: creator-kept object: status ${status}, '${out}'"
for open_obj in "-q 0x00000006 41 someoneelse 666" "-m 0x00000007 14 someoneelse 660 someoneelse $(id -gn)" \
    "-s 0x0000000a 22 someoneelse 660 someoneelse nogroup-x $(id -gn)"; do
  printf '%s\n' "${open_obj}" > "${WORK}/ipc"; : > "${WORK}/log"
  out="$(run_wipe 2>&1)"; status=$?
  [[ "${status}" -ne 0 ]] && grep -q "1 SysV IPC object(s) of another user that ${ME} can read or write" <<< "${out}" \
    && ok "case 4: a foreign SysV object the sandbox user can use (${open_obj%% 0x*}, mode ${open_obj#* * * * }) fails the wipe" \
    || fail "case 4: foreign open object '${open_obj}': status ${status}, '${out}'"
done
printf '%s\n' "-q 0x00000008 42 someoneelse 600" "-m 0x00000009 15 someoneelse 640 someoneelse nogroup-x" > "${WORK}/ipc"; : > "${WORK}/log"
out="$(run_wipe 2>&1)"; status=$?
[[ "${status}" -eq 0 ]] && ok "case 4: foreign SysV objects the sandbox user cannot use are left alone" \
  || fail "case 4: closed foreign objects failed the wipe: status ${status}, '${out}'"
# A failed group lookup must fail the wipe, not skip the group check.
mkdir -p "${WORK}/idbin"
REAL_ID="$(type -P id)"
printf '%s\n' '#!/usr/bin/env bash' \
  'for a in "$@"; do [[ "${a}" == -G || "${a}" == -Gn ]] && exit 1; done' \
  "exec ${REAL_ID} \"\$@\"" > "${WORK}/idbin/id"
chmod +x "${WORK}/idbin/id"
: > "${WORK}/ipc"; : > "${WORK}/log"
out="$(run_wipe PATH="${WORK}/idbin:${WORK}/bin:${PATH}" 2>&1)"; status=$?
[[ "${status}" -ne 0 ]] && grep -qxF 'sandbox-wipe: FAIL: leg-state-not-cleared' <<< "${out}" \
  && ok "case 4: a failed group lookup fails the wipe" \
  || fail "case 4: failed group lookup: status ${status}, '${out}'"
chmod -R u+rwx "${WORK}/shm"
mkdir -p "${WORK}/tmp/keep"; echo secret-prompt > "${WORK}/tmp/keep/secret-prompt-name"
chmod 555 "${WORK}/tmp/keep"
# A changed identity before removal must fail closed and keep the file.
echo 0 > "${WORK}/stat-race-count"
seed_ipc_clean() { : > "${WORK}/ipc"; }
seed_ipc_clean
out="$(run_wipe STUB_RACE_PATH="${WORK}/tmp/keep/secret-prompt-name" \
  STUB_RACE_COUNT="${WORK}/stat-race-count" 2>&1)"; status=$?
[[ "${status}" -ne 0 ]] && grep -q "sandbox-wipe: FAIL: leg-state-not-cleared" <<< "${out}" \
  && ok "case 4: a file that survives fails the wipe" || fail "case 4: status ${status}, '${out}'"
grep -q "secret-prompt" <<< "${out}" && fail "case 4: a surviving file's name was printed" \
  || ok "case 4: only a count is printed, never a name"
chmod -R u+rwx "${WORK}/tmp"

# --- case 5: no sudo to the sandbox user --------------------------------------
out="$(run_wipe STUB_SUDO_FAIL=1 2>&1)"; status=$?
[[ "${status}" -ne 0 ]] && grep -q "cannot wipe" <<< "${out}" \
  && ok "case 5: a failing sudo fails the wipe" || fail "case 5: status ${status}, '${out}'"

# --- case 7: no file-backed state crosses candidate legs ---------------------
JOB_ROOT="${WORK}/sandbox"
JOB="${JOB_ROOT}/job-1"
OUTSIDE="${WORK}/outside-marker"
mkdir -p "${JOB_ROOT}" "${JOB}"
run_wipe SANDBOX_ROOT="${JOB_ROOT}" > "${WORK}/out7-start" 2>&1
status=$?
[[ "${status}" -eq 0 ]] \
  && ok "case 7: job-start shared wipe needs no job directory" \
  || fail "case 7: job-start wipe exit ${status}: $(cat "${WORK}/out7-start")"
for entry in zig-global-cache:2770 zig-cache:2770 out:2770 adapter-home:2770 sock:2750; do
  name="${entry%%:*}"
  mode="${entry##*:}"
  mkdir -m "${mode}" "${JOB}/${name}"
done

# Fake participant leg 1 writes one marker in every path it can keep.
for name in zig-global-cache zig-cache out adapter-home sock; do
  printf 'pair-1 decode answer\n' > "${JOB}/${name}/leg-1-marker"
done
printf 'must survive\n' > "${OUTSIDE}"
ln -s "${OUTSIDE}" "${JOB}/adapter-home/outward-link"

declare -a before=()
for name in zig-global-cache zig-cache out adapter-home sock; do
  before+=("$(stat -c '%u %g %a' "${JOB}/${name}" 2>/dev/null \
    || stat -f '%u %g %Lp' "${JOB}/${name}")")
done
: > "${WORK}/log"
run_wipe SANDBOX_ROOT="${JOB_ROOT}" SANDBOX_JOB=job-1 > "${WORK}/out7" 2>&1
status=$?
if [[ "${status}" -eq 0 ]]; then
  ok "case 7: candidate leg teardown succeeds"
else
  fail "case 7: teardown exit ${status}: $(cat "${WORK}/out7")"
fi

# Fake participant leg 2 searches every surviving writable path.
found="$(find -P "${JOB}" -mindepth 2 \( -name leg-1-marker -o -name outward-link \) -print)"
[[ -z "${found}" ]] \
  && ok "case 7: leg 2 cannot read a marker from any leg 1 writable path" \
  || fail "case 7: leg 2 found retained state: ${found}"
[[ "$(cat "${OUTSIDE}")" == "must survive" ]] \
  && ok "case 7: the outward symlink target was not followed" \
  || fail "case 7: the outward symlink target changed"
for index in 0 1 2 3 4; do
  name="$(printf '%s\n' zig-global-cache zig-cache out adapter-home sock | sed -n "$((index + 1))p")"
  after="$(stat -c '%u %g %a' "${JOB}/${name}" 2>/dev/null \
    || stat -f '%u %g %Lp' "${JOB}/${name}")"
  [[ "${after}" == "${before[${index}]}" ]] \
    || fail "case 7: ${name} identity changed from ${before[${index}]} to ${after}"
done
[[ -z "$(find -P "${JOB}" -mindepth 2 -print -quit)" ]] \
  && ok "case 7: all five paths are empty and keep their owner and mode" \
  || fail "case 7: a per-job path is not empty"

# Case 15: kernel keyrings.
seed; : > "${WORK}/log"
printf '%s\n' 7001 > "${WORK}/keyrings/ring-424242"
printf '%s\n' 7002 7003 > "${WORK}/keyrings/ring-at-u"
printf '%s\n' 7004 > "${WORK}/keyrings/ring-at-us"
printf '%s\n' 7005 > "${WORK}/keyrings/ring-at-s"
run_wipe > "${WORK}/out-keys" 2>&1; status=$?
[[ "${status}" -eq 0 && ! -s "${WORK}/keyrings/ring-424242" && ! -s "${WORK}/keyrings/ring-at-u" \
    && ! -s "${WORK}/keyrings/ring-at-us" && ! -s "${WORK}/keyrings/ring-at-s" ]] \
  && grep -qxF -- "keyctl clear 424242" "${WORK}/log" \
  && grep -qxF -- "keyctl clear @u" "${WORK}/log" \
  && grep -qxF -- "keyctl clear @us" "${WORK}/log" \
  && grep -qxF -- "keyctl clear @s" "${WORK}/log" \
  && grep -q -- "^-n -u ${ME} -- keyctl session - bash -c" "${WORK}/log" \
  && grep -qxF -- "keyctl link @u @s" "${WORK}/log" \
  && ok "case 15: persistent, user, user-session and session keyrings are emptied as the sandbox user in a new session" \
  || fail "case 15: keyrings not emptied (status ${status}): $(cat "${WORK}/out-keys")"
seed; : > "${WORK}/log"
printf '%s\n' 7002 > "${WORK}/keyrings/ring-at-u"
run_wipe STUB_KEY_STUCK=1 > "${WORK}/out-keys-stuck" 2>&1; status=$?
[[ "${status}" -ne 0 ]] && grep -q "kernel keyrings are not empty" "${WORK}/out-keys-stuck" \
    && ! grep -q 7002 "${WORK}/out-keys-stuck" \
  && ok "case 15: a key that stays fails the wipe, with no key id printed" \
  || fail "case 15: a stuck keyring did not fail the wipe cleanly (status ${status})"
: > "${WORK}/keyrings/ring-at-u"
seed; : > "${WORK}/log"
run_wipe STUB_PERSIST=unsupported > "${WORK}/out-keys-nop" 2>&1; status=$?
[[ "${status}" -eq 0 ]] && ! grep -qF -- "keyctl clear 424242" "${WORK}/log" \
    && grep -qxF -- "keyctl clear @u" "${WORK}/log" \
  && ok "case 15: a kernel without persistent keyrings passes, the other keyrings are still emptied" \
  || fail "case 15: unsupported persistent keyrings gave status ${status}: $(cat "${WORK}/out-keys-nop")"
seed; : > "${WORK}/log"
run_wipe STUB_PERSIST=fail > "${WORK}/out-keys-pfail" 2>&1; status=$?
[[ "${status}" -ne 0 ]] && grep -q "kernel keyrings are not empty" "${WORK}/out-keys-pfail" \
  && ok "case 15: any other get_persistent failure fails the wipe" \
  || fail "case 15: a failed get_persistent passed (status ${status})"

if [[ "${failures}" -eq 0 ]]; then
  echo "PASS: test-sandbox-wipe.sh"
  exit 0
fi
echo "FAIL: ${failures} failure(s)" >&2
exit 1
