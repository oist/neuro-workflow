# Community Jupyter tree

This directory is the **community** Lab filesystem (Hub user `community`,
live alias `hackathon`). It is mounted instead of `codes/` for that Lab.

```
codes-hackathon/   (live name; codes-community/ is the canonical name if present)
  projects/   # community FlowProject dirs (UUID)
  nodes/      # tenant-scoped node files (not the project catalog)
```

The neuroworkflow Python library is still mounted read-only from
`codes/neuroworkflow` into the community container.

Do **not** copy the project `codes/nodes` catalog here automatically.
Public community nodes are an explicit allow-list / copy at cutover.

See `deployment/JUPYTER_TWO_SPACES.md`.
