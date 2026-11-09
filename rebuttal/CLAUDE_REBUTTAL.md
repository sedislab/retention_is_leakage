# Working rules for the rebuttal phase

Current task: the ICLR 2027 rebuttal experiments on Kodiak. At the start of every session, read
`rebuttal/STATE.md` and `rebuttal/REBUTTAL_EXPERIMENTS.md`, then continue from "Next action".

- Cluster: Baylor Kodiak, OpenPBS, queue `batch`, at most 512 concurrent cores. Work under `/data`,
  never on the login node.
- Never modify, overwrite or delete existing files in `results/`, `figs/`, `tables/`, `features/`
  or `shadows/`. New outputs are `results/rb_*.csv` with provenance sidecars, and new per-shadow
  files go under `shadows_rb/`.
- Do not edit existing source files, configs or job templates. Add new files only.
- Choose hyperparameters, projections and clip norms only on public `ref` data.
- Fix every protocol before running it. Report failed predictions as they are.
- Stop and ask the user at every gate, and before any job above 2,000 CPU core-hours.
- Update `rebuttal/STATE.md` after every meaningful step.
- These rules add to the repository's existing CLAUDE.md, if there is one. They do not replace it.
