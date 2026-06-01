# Cursor → Remote Slurm Cluster Manual (Reusable, Non-Interactive)

## What is already set up
- Local launcher: `~/remote_srun.sh`
- Remote runner: `~/remote_run.sh` (on your cluster login host)
- Non-interactive SSH to your cluster alias is working.

---

## 1) One-time env defaults (recommended)
Create local defaults file:

```bash
cat > ~/.remote_srun.env <<'EOF'
export REMOTE_HOST=<CLUSTER_SSH_ALIAS>
export SLURM_PARTITION=<PARTITION_NAME>
export SLURM_GRES=<GPU_REQUEST>          # e.g. gpu:1 or gpu:a100:1
export SLURM_CPUS=<CPU_COUNT>            # e.g. 8
export SLURM_MEM=<MEMORY>                # e.g. 32G
export SLURM_TIME=<HH:MM:SS>             # e.g. 24:00:00

# choose ONE env setup style:
# Conda example:
export REMOTE_RUN_SETUP='source ~/miniconda3/etc/profile.d/conda.sh && conda activate <ENV_NAME>'

# Venv example (instead of conda):
# export REMOTE_RUN_SETUP='source <REMOTE_PROJECT_DIR>/.venv/bin/activate'
EOF
```

Reload shell:

```bash
source ~/.remote_srun.env
```

---

## 2) Basic usage
Dry-run first:

```bash
~/remote_srun.sh --dry-run <REMOTE_PROJECT_DIR> python -V
```

Run:

```bash
~/remote_srun.sh <REMOTE_PROJECT_DIR> python -V
```

Train:

```bash
~/remote_srun.sh --log <REMOTE_PROJECT_DIR> python train.py --exp <EXP_NAME>
```

Eval:

```bash
~/remote_srun.sh --log <REMOTE_PROJECT_DIR> python eval.py --checkpoint <CKPT>
```

---

## 3) GitHub + auto-pull workflow (no separate terminal)
Use these flags per run:

- `--github-test` → runs `ssh -T git@github.com` on remote (non-blocking)
- `--git-pull` → runs `git pull --ff-only` in remote project before command

Example:

```bash
~/remote_srun.sh --github-test --git-pull --log <REMOTE_PROJECT_DIR> python train.py --exp <EXP_NAME>
```

---

## 4) Useful checks
Queue/jobs:

```bash
ssh <CLUSTER_SSH_ALIAS> 'squeue -u <REMOTE_USERNAME>'
```

Recent logs:

```bash
ssh <CLUSTER_SSH_ALIAS> 'ls -lt <REMOTE_PROJECT_DIR>/logs | head'
```

Tail newest log:

```bash
ssh <CLUSTER_SSH_ALIAS> 'LATEST=$(ls -t <REMOTE_PROJECT_DIR>/logs/run_*.out | head -n1); echo "$LATEST"; tail -n 200 "$LATEST"'
```

---

## 5) Cursor instruction block (paste into Cursor chat)

```text
You are operating in a local repo, but all heavy runs must go through my remote launcher.

Rules:
1) NEVER run training/eval directly on local machine.
2) ALWAYS use ~/remote_srun.sh for remote execution.
3) ALWAYS do a dry-run first for any new command.
4) For long runs, include --log.
5) For synced runs, include --github-test --git-pull.
6) NEVER delete files unless I explicitly say so.

Execution template:
- Dry run:
  ~/remote_srun.sh --dry-run <REMOTE_PROJECT_DIR> <COMMAND>
- Real run:
  ~/remote_srun.sh --github-test --git-pull --log <REMOTE_PROJECT_DIR> <COMMAND>

After launching, show me:
- exact command used
- whether srun allocated
- hostname and working directory from remote_run output
- where logs are written
```

---

## 6) Notes
- If auth regresses, verify: `ssh -o BatchMode=yes <CLUSTER_SSH_ALIAS> 'echo ok'`.
- Keep runs non-interactive and script-first for overnight reliability.
