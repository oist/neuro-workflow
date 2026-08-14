# Hackathon Jupyter tree

This directory is the **hackathon** Lab filesystem (Hub user `hackathon`).
It is mounted instead of `codes/` for that Lab.

```
codes-hackathon/
  projects/   # hackathon FlowProject dirs (UUID)
  nodes/      # tenant-scoped node files (not the internal catalog)
```

The neuroworkflow Python library is still mounted read-only from
`codes/neuroworkflow` into the hackathon container.

Do **not** copy the internal `codes/nodes` catalog here automatically.
Public hackathon nodes are an explicit allow-list / copy at cutover.

See `deployment/JUPYTER_TWO_SPACES.md`.
