# Execution boundary

Participant code runs without network access.

The benchmark owns artifact paths and resident sockets.

The target and drafter snapshots are read-only.

Each leg gets a new resident process.

The benchmark stops that process after the leg.

The runner must isolate the participant process from runner credentials.

## Operator setup

Run the one-time setup on each ranked Linux box:

```bash
sudo tools/box-runner/setup-sandbox-user.sh
sudo tools/box-runner/setup-sandbox-user.sh --check
```

The first command creates the sandbox account, shared group, writable sandbox
root, network-deny rules, ACL services, and wipe list. The second command
checks that boundary without running a benchmark.

The ranked runner service must export the staged asset paths named in
`.github/workflows/benchmark.yml`, including the target snapshot, golden
directory, reference workspace, and box calibration file. Run
`tools/ranked-box-preflight.sh` before a scored job.

## Sandbox names

The setup scripts use these defaults. They are local account and group names.
The runner service can set other names in its environment.

| Variable | Default | Purpose |
|---|---|---|
| `RUNNER_USER` | `runner` | Account that owns the actions runner |
| `SANDBOX_USER` | `qwen38sub` | Unprivileged participant account |
| `SHARE_GROUP` | `qwen38share` | Read-only handoff group |
| `SANDBOX_ROOT` | `/var/lib/qwen38/sandbox` | Per-job sandbox root |
| `SANDBOX_ETC` | `/etc/qwen38-sandbox` | Root-owned sandbox configuration |

A second runner on the same box must use distinct sandbox names and paths.
