# CI coverage

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
