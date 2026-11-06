# CI coverage

CI runs `.github/workflows/ci.yml` on every pull request, on every push to
`main`, and when dispatched by hand. Its `lint` job runs on Linux:
`ubuntu-latest`, or a Darkbloom Linux Sandbox (see "Runner routing").

CI parses both JSON contracts.

CI runs shell syntax and warning-level shell checks outside the vendored
engine. It then runs the selected control-plane tests named in
`.github/workflows/ci.yml`, including manifest lint, benchd platform checks,
document consistency, editable-surface review, resident lifecycle, memory
planning, setup onboarding, and the local baseline entry point.

CI checks the editable surface.

CI does not build the native TensorFold worker. The ranked Linux GB10 path
builds it through `./setup.sh`.

Real model correctness needs the ranked GB10 box.

## Runner routing

`.github/workflows/darkbloom-route.yml` chooses where the `lint` job runs: a
Darkbloom Sandbox when the Darkbloom plane grants one, otherwise its
GitHub-hosted fallback. The route job runs on GitHub-hosted `ubuntu-24.04`
with no token permissions and no checkout. The `lint` job runs on
`fromJSON(needs.route.outputs.routes)['lint']`. The lane is the job name.

| Lane | Darkbloom profile | Fallback |
|---|---|---|
| `lint` | `linux_arm64` | `ubuntu-latest` |

The `privileged` job runs `tools/test-sandbox-wipe.sh` on `ubuntu-latest`
only. It is never routed: the test needs passwordless sudo to a second user
and a supplementary group, and a Darkbloom runner gives a job neither.

Routing is off unless the repository variable `DARKBLOOM_ROUTING` is exactly
`on`. While it is off, and for pull requests from forks, the job keeps its
fallback and the route job sends no request. When it is on, the route job
sends one `POST /v1/ci/routes` to `DARKBLOOM_SANDBOX_URL` (default
`https://api-sandbox.darkbloom.dev`). The request names only the repository,
the run ID, the run attempt and the lanes. It holds no credential. Errors,
timeouts and malformed replies keep the fallback. The plane never fails the
route job.

Darkbloom Linux Sandboxes are ARM64; the `ubuntu-latest` fallback is x64. The
`lint` job installs the pinned actionlint and ShellCheck assets that match the
runner. The decision is fixed for the run: **Re-run failed jobs** reuses it.
To move a run back to GitHub-hosted runners, turn routing off and use
**Re-run all jobs**.

The ranked pipeline, `.github/workflows/benchmark.yml`, is not routed.
