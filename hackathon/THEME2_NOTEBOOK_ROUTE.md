# Theme 2, notebook route — build your workflow without leaving the web app

> Part of the **NeuroWorkflow Hackathon** kit. [`PARTICIPANT_GUIDE.md`](./PARTICIPANT_GUIDE.md)
> describes Theme 2 with a *local* coding agent (Claude Code / Codex) and
> [`README.md`](./README.md) is its step-by-step walkthrough. **This file describes the
> alternative route: doing Theme 2 entirely inside the NeuroWorkflow web app.**

The chat agent that runs **inside JupyterLab** on the server can read the code you brought,
write the node classes, and add the nodes and edges to your project — no local agent setup,
no separate upload step for the workflow itself.

## What you need

- An account on the NeuroWorkflow web app and a **project** (create one in the GUI if you
  don't have one yet).
- Your source code (a NEST/TVB/Brian2 script, a tutorial notebook, …). Unstructured is fine.

## Steps

1. In the web app, select your project and click **Open JupyterLab Tab**. Keep this tab
   open while you work — it is what authorizes the agent to edit *your* project.
2. In JupyterLab you land in your project folder (`codes/projects/<project id>/`). Upload
   your script there (drag & drop) and create a new **Python 3** notebook in the same folder.
3. In the notebook run:

   ```python
   %load_ext neuroworkflow.agent
   from neuroworkflow.agent import ChatPanel
   ChatPanel()
   ```

   A chat panel appears. If it prints a warning about workflow tools being disabled, go back
   to the app tab, make sure the project's Jupyter tab is open, and re-run the cell.

4. Ask the agent, for example:

   > Read `run_simulations.py` in this folder and propose how to split it into NeuroWorkflow
   > nodes. Ask me to confirm the breakdown before writing anything. Then create the nodes,
   > add them and their connections to this project's workflow, regenerate the workflow code,
   > and run it here so we can check the outputs.

   The agent follows the `create-node` conventions, writes nodes under
   `/home/jovyan/codes/nodes/sandbox/`, and uses the project's workflow tools
   (`add_node`, `add_edge`, `update_node_parameter`, `generate_code_batch`, …). When it is
   done, reload the project in the GUI to see the graph. If a new node type does not show up
   in the app's node palette, ask the agent to upload (register) the node file in the app as
   well.

   You can also chat inline with `%chat <your question>`.

## Tips

- The agent only operates on **the project whose folder your notebook is in**; to work on
  another project, open that project's Jupyter tab and a notebook in its folder.
- Confirm the proposed node breakdown before the agent writes code — it is the most
  important decision.
- Keep simulations **lightweight** (short runs, small networks): everything runs on the
  application server.
- Closing the app tab (or a long break) can expire the agent's access; reopen the project's
  Jupyter tab and just continue chatting — no restart needed.
- Anything the agent writes lives in your project folder and `codes/nodes/sandbox/`; check
  the generated files into your own repository if you want to keep them.
