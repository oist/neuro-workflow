# RIKEN CBS data download — spike

**Status: exploratory. Nothing is implemented yet, and the design is not settled.**
This branch exists so the work has a home while it is being scoped. Please do not
build on it or open PRs against it.

## Goal

Connect NeuroWorkflow to the remote RIKEN CBS data repository and fetch a folder of
data onto the app server, so it can be used from a workflow.

## To be decided with the RIKEN CBS data manager

1. **Transfer method and credentials** — Aspera, S3, SFTP, or an HTTP API. This decides
   everything else. The nest kernel image already carries the Aspera client and the AWS
   CLI, so those two need no new tooling.
2. **Where the folder lands** — inside a project (`codes/projects/<id>/`, so it appears in
   the GUI and in the Lab), or a shared data area outside any project.
3. **Who may fetch it** — after PR #88 the app has two spaces (project and community), so
   a download endpoint needs to say which of them may reach this repository.

## Deliberately out of scope for now

GUI work, a REST endpoint, scheduling and caching. First establish that the transfer
works at all, by hand, with the data manager watching.
