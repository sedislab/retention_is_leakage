# 05 — RUNBOOK: running this on Kodiak

One bootstrap, then **one prompt**. Claude Code runs the rest autonomously and stops only at phase
gates to ask whether to continue.

---

## 1. Bootstrap (shell, on Kodiak)

```bash
ssh islamm@kodiak.baylor.edu
passwd                                    # if you have not already rotated the initial password

mkdir -p /data/islamm/retention_leakage/{raw,datasets,features,shadows,runs,logs,envs}

module avail 2>&1 | grep -i node          # load a nodejs module if one exists
node --version                            # need v18+
# if not available, install nvm under /data (the login node has internet):
export NVM_DIR=/data/islamm/.nvm && mkdir -p "$NVM_DIR"
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
source "$NVM_DIR/nvm.sh" && nvm install 22

npm install -g @anthropic-ai/claude-code && claude --version

cat >> ~/.bashrc <<'EOS'
export NVM_DIR=/data/islamm/.nvm
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
export P3=/data/islamm/retention_leakage
export HF_HOME=$P3/hf TORCH_HOME=$P3/torch
EOS
```

Refresh the specs from **your Mac** whenever they change (there is no code to copy — Claude Code writes all of it):

```bash
rsync -avz --exclude '__pycache__' --exclude '.DS_Store' \
  ~/Desktop/Research/CFL/P3_LifelongPrivacy_FCL/ \
  islamm@kodiak.baylor.edu:/data/islamm/retention_leakage/
```

The repo is already on Kodiak at `/data/islamm/retention_leakage` — nothing to clone, nothing to
initialise. **No git required.** Provenance is stamped with a content hash of `code/`, not a commit.

### Let it work without interrupting you

Autonomy dies if Claude Code stops for approval every thirty seconds. Pre-authorise the commands it
needs — create `.claude/settings.local.json` in the repo:

```json
{
  "permissions": {
    "allow": [
      "Bash(qsub:*)", "Bash(qstat:*)", "Bash(qdel:*)", "Bash(pbsnodes:*)",
      "Bash(module:*)", "Bash(python:*)", "Bash(python3:*)", "Bash(pip:*)", "Bash(pytest:*)",
      "Bash(make:*)", "Bash(git:*)", "Bash(ls:*)", "Bash(cat:*)", "Bash(grep:*)", "Bash(find:*)",
      "Bash(mkdir:*)", "Bash(cp:*)", "Bash(mv:*)", "Bash(tar:*)", "Bash(unzip:*)",
      "Bash(wget:*)", "Bash(curl:*)", "Bash(du:*)", "Bash(df:*)", "Bash(nvidia-smi:*)",
      "Edit", "Write", "Read", "Glob", "Grep"
    ]
  }
}
```

`claude --permission-mode acceptEdits` is the lighter-touch alternative. `--dangerously-skip-permissions`
also works but removes every guardrail on a shared cluster — if you use it, the login-node rule in
`CLAUDE.md` is the only thing keeping heavy work off the login node, so watch for it.

```bash
cd /data/islamm/retention_leakage && claude
```

---

## 2. The prompt

Paste this once. That is the whole interaction until the first gate.

> Read `CLAUDE.md` first — it contains the autonomy contract that governs how you work. Then read
> everything in `build/` (00 through 06 and STATE.md), then `RESEARCH_PLAN.md` sections 0–5 and
> `agents/OPEN_QUESTIONS.md` in full.
>
> You are building this entire system from scratch. There is no prior code — `build/06_PACKAGE_SPEC.md`
> is the source of truth for the abstractions. Work through `build/00_BUILD_PLAN.md` phases P0 → P9 in
> order, and **work autonomously**: do not ask me to approve individual files, jobs, downloads or
> design choices. Install what you need, download the datasets, write the PBS scripts, submit the
> jobs, poll them, debug failures, generate every figure and table in `build/03_RESULTS_SPEC.md` with
> its CSV, and keep going.
>
> **Stop only at a phase gate.** When one passes, show me what was built, the gate numbers, and
> anything that surprised you, then ask "proceed to P*n+1*?" and wait. The other three stopping
> conditions are in `CLAUDE.md` — a gate failing twice for the same reason, something that would
> relitigate a `[LOCKED]` decision, and setting a hypothesis to CONFIRMED. Nothing else.
>
> Constraints that are not negotiable, all of them in the build files but worth repeating because
> they are the ones that erode: you are on the **login node** — edit and `qsub` only, never train or
> extract in-process. The artifact **ledger is the only interface** between methods and attacks.
> `touched` gets over-reported, never under-reported. **No number reaches the paper except through a
> CSV** produced by a logged run. Shadow arrays must fit the **512 concurrent-core cap**
> (`ncpus=8`, `-J 0-N%60`). Reimplement published methods inside our harness from their papers —
> **do not clone their repos**, time-box each to one session, record the gap in TAB07. Negative
> results get reported, not dropped.
>
> **Never idle**: when a job is queued, switch to work that does not depend on it — `00_BUILD_PLAN.md`
> says which phases overlap. And keep `build/STATE.md` current as you go, because your context will
> be compacted during a run this long and that file plus `results/RUN_LOG.jsonl` is how the next
> session resumes.
>
> Start with P0. Submit the GPU probe job first so it queues while you work, then begin building
> `artifacts.py`.

At each gate you reply with one word. `continue`, or `stop, <what you want changed>`.

---

## 3. Resuming after a context reset

Long runs get compacted. To restart a session cleanly:

```bash
cd /data/islamm/retention_leakage && claude
```

> Read `CLAUDE.md` and `build/STATE.md`, then `results/RUN_LOG.jsonl` and `qstat -u islamm`.
> Tell me where the project actually stands, then continue from STATE.md's "Next action" under the
> same autonomy contract. Do not redo finished work — check the run log before recomputing anything.

---

## 4. Watching it from the outside

```bash
qstat -u islamm                                   # your jobs
qstat -u islamm -t | head -40                     # array subjobs
tail -f /data/islamm/retention_leakage/logs/*.o*              # live job output
cat /data/islamm/retention_leakage/build/STATE.md        # where it thinks it is
tail -5 /data/islamm/retention_leakage/results/RUN_LOG.jsonl
du -sh /data/islamm/retention_leakage/*                       # the shadow store is what grows
ls /data/islamm/retention_leakage/figs/                  # figures as they land
```

A result with no line in `RUN_LOG.jsonl` did not happen. If it reports a number you cannot find
there, ask where the log line is.

---

## 5. Failure playbook

| Symptom | What to do |
|---|---|
| Probe job gets no GPU under `-q gpu -l select=1:ncpus=8:ngpus=1` | Email HPRCS@baylor.edu for the right incantation. Meanwhile everything cacheable proceeds — that is most of the project. |
| Array job will not run at the requested concurrency | The `batch` per-user cap is 512 cores. Lower `%K` before lowering the shadow count, and check the shadow-count plateau plot first. |
| Compute nodes have no internet | The two-phase `--fetch`/`--prepare` design already covers this. One rate-limited login-node stream at a time — never parallel. |
| `/data` filling up | Shadows are the bulk. Get the real quota from HPRCS. Cut shadow count per config before cutting a dataset. |
| It starts running training on the login node | Stop it and point at the autonomy contract in `CLAUDE.md`. This is the instruction most likely to erode over a long session. |
| It asks permission for routine things | The allow-list in §1 is missing an entry. Add it rather than approving each time. |
| It wants to relitigate a `[LOCKED]` decision | It is supposed to stop and ask. Decide yourself; do not let it rewrite the plan. |
| A gate fails twice the same way | It should have stopped. If it did not, stop it, and have it write the problem into `notes/`. |
