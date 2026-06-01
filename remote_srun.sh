#!/usr/bin/env bash
set -euo pipefail

# Reusable non-interactive SLURM launcher over SSH.
# Safe defaults: read-only orchestration, no deletions.

# Optional local defaults file (no-op if missing):
#   ~/.remote_srun.env
# Example vars:
#   export REMOTE_HOST=athena
#   export SLURM_PARTITION=gpu
#   export SLURM_GRES=gpu:a30:1
#   export SLURM_CPUS=8
#   export SLURM_MEM=32G
#   export SLURM_TIME=24:00:00
#   export REMOTE_RUN_SETUP='source ~/miniconda3/etc/profile.d/conda.sh && conda activate myenv'
if [[ -f "$HOME/.remote_srun.env" ]]; then
  # shellcheck disable=SC1090
  source "$HOME/.remote_srun.env"
fi

HOST="${REMOTE_HOST:-athena}"
PARTITION="${SLURM_PARTITION:-gpu}"
GRES="${SLURM_GRES:-gpu:a30:1}"
CPUS="${SLURM_CPUS:-8}"
MEM="${SLURM_MEM:-32G}"
TIME="${SLURM_TIME:-24:00:00}"
REMOTE_RUNNER="${REMOTE_RUNNER:-~/remote_run.sh}"
SETUP_CMD="${REMOTE_RUN_SETUP:-}"
LOG_DIR="${REMOTE_LOG_DIR:-}"
LOG_MODE=0
DRY_RUN=0
BOOTSTRAP_REMOTE=0
GIT_PULL=0
GITHUB_TEST=0

usage() {
  cat <<USAGE
Usage:
  remote_srun.sh [options] <remote_project_dir> <command...>

Options:
  --host <h>              SSH host alias (default: ${HOST})
  --partition <p>         SLURM partition (default: ${PARTITION})
  --gres <g>              SLURM gres (default: ${GRES})
  --cpus <n>              SLURM cpus-per-task (default: ${CPUS})
  --mem <m>               SLURM memory (default: ${MEM})
  --time <t>              SLURM time (default: ${TIME})
  --setup '<cmd>'         Remote env setup (conda/venv activation) before command
  --log-dir <dir>         Remote directory for log files (default: <project>/logs)
  --git-pull              Run git pull --ff-only in project before command
  --github-test           Run ssh -T git@github.com check before git pull
  --log                   Save logs to <project>/logs/run_<timestamp>.out
  --dry-run               Print remote command, do not execute
  --bootstrap-remote      Create/update ~/remote_run.sh on remote host
  -h, --help              Show this help

Examples:
  ./remote_srun.sh --setup 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate trak' ~/trak-traj python train.py
  ./remote_srun.sh --git-pull --github-test --log ~/trak-traj python eval.py --checkpoint best.pt
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    --partition) PARTITION="$2"; shift 2 ;;
    --gres) GRES="$2"; shift 2 ;;
    --cpus) CPUS="$2"; shift 2 ;;
    --mem) MEM="$2"; shift 2 ;;
    --time) TIME="$2"; shift 2 ;;
    --setup) SETUP_CMD="$2"; shift 2 ;;
    --log-dir) LOG_DIR="$2"; shift 2 ;;
    --git-pull) GIT_PULL=1; shift ;;
    --github-test) GITHUB_TEST=1; shift ;;
    --log) LOG_MODE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --bootstrap-remote) BOOTSTRAP_REMOTE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; break ;;
    -*) echo "Unknown option: $1" >&2; usage; exit 2 ;;
    *) break ;;
  esac
done

if [[ $# -lt 2 ]]; then
  usage
  exit 2
fi

PROJECT_DIR="$1"
shift

# If user passed a local home path, map to remote ~ for portability
if [[ "${PROJECT_DIR}" == "${HOME}/"* ]]; then
  PROJECT_DIR="~/${PROJECT_DIR#${HOME}/}"
fi

# Safely quote user command for remote shell
printf -v USER_CMD '%q ' "$@"
USER_CMD="${USER_CMD% }"

if [[ "$BOOTSTRAP_REMOTE" -eq 1 ]]; then
  ssh "$HOST" 'cat > ~/remote_run.sh <<'"'"'EOS'"'"'
#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$1"
shift

source ~/.bashrc || true
cd "$PROJECT_DIR" || { echo "[remote_run] cannot cd to $PROJECT_DIR" >&2; exit 1; }

if [[ -n "${REMOTE_RUN_SETUP:-}" ]]; then
  echo "[remote_run] applying setup..."
  eval "${REMOTE_RUN_SETUP}"
fi

if [[ "${REMOTE_GITHUB_TEST:-0}" == "1" ]]; then
  echo "[remote_run] github ssh test..."
  ssh -o BatchMode=yes -o ConnectTimeout=8 -T git@github.com || true
fi

if [[ "${REMOTE_GIT_PULL:-0}" == "1" ]]; then
  if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "[remote_run] git pull --ff-only"
    git pull --ff-only
  else
    echo "[remote_run] skip git pull (not a git repo)"
  fi
fi

echo "================================="
echo "Running on: $(hostname)"
echo "Working directory: $(pwd)"
echo "Command: $*"
echo "Started at: $(date -Is)"
echo "================================="

exec "$@"
EOS
chmod +x ~/remote_run.sh'
  echo "[ok] bootstrapped ~/remote_run.sh on ${HOST}"
fi

# Quote env vars for safe remote interpolation
printf -v Q_SETUP '%q' "$SETUP_CMD"

# GitHub test + git pull run on login host BEFORE srun (more reliable than compute nodes)
PREP_CMD=""
if [[ "$GITHUB_TEST" -eq 1 ]]; then
  PREP_CMD+="echo '[preflight] github ssh test...'; /usr/bin/ssh -o BatchMode=yes -o ConnectTimeout=8 -T git@github.com || true; "
fi
if [[ "$GIT_PULL" -eq 1 ]]; then
  PREP_CMD+="echo '[preflight] git pull --ff-only in ${PROJECT_DIR}'; cd ${PROJECT_DIR} && /usr/bin/git pull --ff-only; "
fi

REMOTE_ENV="REMOTE_RUN_SETUP=${Q_SETUP} REMOTE_GIT_PULL=0 REMOTE_GITHUB_TEST=0"
BASE_SRUN="${REMOTE_ENV} srun -p ${PARTITION} --gres=${GRES} --cpus-per-task=${CPUS} --mem=${MEM} --time=${TIME} bash ${REMOTE_RUNNER} ${PROJECT_DIR} ${USER_CMD}"

if [[ "$LOG_MODE" -eq 1 ]]; then
  REMOTE_LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/logs}"
  REMOTE_CMD="mkdir -p ${REMOTE_LOG_DIR} && ${PREP_CMD}${BASE_SRUN} > ${REMOTE_LOG_DIR}/run_\$(date +%s_%N).out 2>&1"
else
  REMOTE_CMD="${PREP_CMD}${BASE_SRUN}"
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "ssh ${HOST} \"${REMOTE_CMD}\""
  exit 0
fi

exec ssh "$HOST" "$REMOTE_CMD"
