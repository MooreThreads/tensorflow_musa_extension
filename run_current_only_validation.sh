#!/usr/bin/env bash

# Current-only validation runner based on .github/workflows/pr-validation.yml.
#
# This script intentionally does not fetch, checkout, clean, or build source.
# Build the current wheel yourself first, then run this script from the repo root.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON:-python3}"
LOG_BASE="${LOG_BASE:-$REPO_ROOT/current_only_logs}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-$LOG_BASE/current/$RUN_ID}"
MODEL_ROOT="${MODEL_ROOT:-/home/runner/tf_test_model_wheel}"
WHEEL_PATH="${WHEEL_PATH:-}"

INTEGRATION_TIMEOUT="${INTEGRATION_TIMEOUT:-60m}"
FUSION_TIMEOUT="${FUSION_TIMEOUT:-60m}"
T_PERF_TIMEOUT="${T_PERF_TIMEOUT:-20m}"
T_ACCURACY_TIMEOUT="${T_ACCURACY_TIMEOUT:-20m}"
BD_MODEL_TIMEOUT="${BD_MODEL_TIMEOUT:-30m}"
BD_MODEL_BS="${BD_MODEL_BS:-32,128,256,1024}"
BD_MODEL_RUN_ITERS="${BD_MODEL_RUN_ITERS:-30}"

ONETRANS_PERF_ROOT="${ONETRANS_PERF_ROOT:-$MODEL_ROOT/training/onetrans}"
ONETRANS_PERF_SCRIPT_NAME="${ONETRANS_PERF_SCRIPT_NAME:-train_perf_bf16.py}"
ONETRANS_PERF_DATA_DIR="${ONETRANS_PERF_DATA_DIR:-$ONETRANS_PERF_ROOT/data}"
ONETRANS_PERF_WARMUP_STEPS="${ONETRANS_PERF_WARMUP_STEPS:-20}"
ONETRANS_PERF_MEASURE_STEPS="${ONETRANS_PERF_MEASURE_STEPS:-50}"
ONETRANS_PERF_TIMEOUT="${ONETRANS_PERF_TIMEOUT:-60m}"

TOKENLARGE_PERF_ROOT="${TOKENLARGE_PERF_ROOT:-$MODEL_ROOT/training/tokenmixer-large}"
TOKENLARGE_PERF_SCRIPT_NAME="${TOKENLARGE_PERF_SCRIPT_NAME:-train_perf_bf16.py}"
TOKENLARGE_PERF_DATA_DIR="${TOKENLARGE_PERF_DATA_DIR:-$TOKENLARGE_PERF_ROOT/data}"
TOKENLARGE_PERF_WARMUP_STEPS="${TOKENLARGE_PERF_WARMUP_STEPS:-20}"
TOKENLARGE_PERF_MEASURE_STEPS="${TOKENLARGE_PERF_MEASURE_STEPS:-50}"
TOKENLARGE_PERF_TIMEOUT="${TOKENLARGE_PERF_TIMEOUT:-60m}"

TRAINING_TIMEOUT="${TRAINING_TIMEOUT:-180m}"
TRAINING_MODEL_TIMEOUT_SECONDS="${TRAINING_MODEL_TIMEOUT_SECONDS:-3600}"
TRAINING_EPOCHS="${TRAINING_EPOCHS:-100}"
TRAINING_MODE="${TRAINING_MODE:-pr-smoke}"
PR_TRAINING_MODELS="rankmixer"
DAILY_TRAINING_MODELS="rankmixer onetrans tokenmixer-large afm autoint bst ccpm dcn dcnmix deepfefm deepfm dien difm din dsin edcn esmm fgcnn fibinet flen fnn fwfm ifm mlr mmoe nfm onn ple pnn wdl wukong xdeepfm"
TRAINING_MODELS="${TRAINING_MODELS:-}"

ALL_CURRENT_ONLY_JOBS="integration fusion t_accuracy t_perf bd_model1 bd_model2 bd_model3 onetrans_perf tokenlarge_perf training"
DEFAULT_CURRENT_ONLY_JOBS="$ALL_CURRENT_ONLY_JOBS"
CURRENT_ONLY_JOBS="${CURRENT_ONLY_JOBS:-$DEFAULT_CURRENT_ONLY_JOBS}"
FAIL_FAST="${FAIL_FAST:-0}"

EMPTY_TEST_RESULT_PATTERN="${EMPTY_TEST_RESULT_PATTERN:-No tests found|No test files found|Total Tests:[[:space:]]*0|Total Tests[[:space:]]*\\|[[:space:]]*0|no average_time_summary entries|report path missing|log not found}"
MIN_INTEGRATION_TESTS="${MIN_INTEGRATION_TESTS:-500}"

mkdir -p "$LOG_ROOT/summaries"

declare -a SUMMARY_ROWS=()

usage() {
  cat <<'USAGE'
Usage:
  ./run_current_only_validation.sh

Required before running:
  Build the current wheel yourself. The script installs the wheel from:
    1. WHEEL_PATH, if set
    2. ./dist/tensorflow_musa-*.whl, latest by name

Common environment variables:
  WHEEL_PATH=/path/to/tensorflow_musa-*.whl
  MODEL_ROOT=/home/runner/tf_test_model_wheel
  GPU_LOCK_FILE=/tmp/tensorflow_musa_gpu.lock
  CURRENT_ONLY_JOBS="integration t_accuracy t_perf bd_model1"
  CURRENT_ONLY_JOBS=all
  FAIL_FAST=1

Supported jobs:
  integration
  fusion
  t_accuracy
  t_perf
  bd_model1
  bd_model2
  bd_model3
  onetrans_perf
  tokenlarge_perf
  training

By default this script runs all supported current-only jobs.
This script does not fetch, checkout, clean, build, or run baseline.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

log_info() {
  printf '[INFO] %s\n' "$*"
}

log_error() {
  printf '[ERROR] %s\n' "$*" >&2
}

append_summary() {
  local name="$1"
  local status="$2"
  local detail="${3:-}"
  SUMMARY_ROWS+=("$name|$status|$detail")
}

resolve_wheel_path() {
  if [[ -n "$WHEEL_PATH" ]]; then
    if [[ ! -f "$WHEEL_PATH" ]]; then
      log_error "WHEEL_PATH does not exist: $WHEEL_PATH"
      return 1
    fi
    printf '%s\n' "$WHEEL_PATH"
    return 0
  fi

  local wheel
  wheel="$(find "$REPO_ROOT/dist" -maxdepth 1 -type f -name 'tensorflow_musa-*.whl' 2>/dev/null | sort | tail -n 1 || true)"
  if [[ -z "$wheel" ]]; then
    log_error "No wheel found. Set WHEEL_PATH or put tensorflow_musa-*.whl under $REPO_ROOT/dist."
    return 1
  fi
  printf '%s\n' "$wheel"
}

run_logged() {
  local name="$1"
  local log_file="$2"
  shift 2

  mkdir -p "$(dirname "$log_file")"
  log_info "Running $name"
  log_info "Log: $log_file"

  "$@" 2>&1 | tee "$log_file"
  local exit_code=${PIPESTATUS[0]}

  if [[ "$exit_code" == "0" ]]; then
    log_info "$name: success"
  else
    log_error "$name: failure, exit_code=$exit_code"
  fi
  return "$exit_code"
}

run_with_optional_gpu_lock() {
  if [[ -n "${GPU_LOCK_FILE:-}" ]]; then
    mkdir -p "$(dirname "$GPU_LOCK_FILE")"
    (
      exec 9>"$GPU_LOCK_FILE"
      flock 9
      echo "Acquired GPU test lock: $GPU_LOCK_FILE"
      "$@"
    )
  else
    echo "GPU_LOCK_FILE is not set; running without a global GPU lock."
    "$@"
  fi
}

install_current_wheel_body() {
  local wheel="$1"

  "$PYTHON_BIN" -m pip install "$wheel" --no-deps --force-reinstall || return $?
  "$PYTHON_BIN" - <<'PY'
import pathlib
import tensorflow_musa

package_root = pathlib.Path(tensorflow_musa.__file__).resolve().parent
print(f'tensorflow_musa wheel: {package_root}')
PY
}

install_current_wheel() {
  local wheel="$1"
  local log_file="$LOG_ROOT/install/install_current_wheel.log"

  run_logged "install current wheel" "$log_file" install_current_wheel_body "$wheel"
}

job_integration_body() {
  cd "$REPO_ROOT/test" || return $?
  timeout "$INTEGRATION_TIMEOUT" "$PYTHON_BIN" test_runner.py --quiet
}

job_integration() {
  local log_file="$LOG_ROOT/integration/integration_current.log"
  run_logged "integration current" "$log_file" job_integration_body
  local exit_code=$?

  local detail=""
  if [[ -f "$log_file" ]]; then
    local total_tests
    total_tests="$(grep -E "Total Tests[[:space:]]*(\\||:)[[:space:]]*[0-9]+" "$log_file" | tail -1 | grep -oE "[0-9]+" | tail -1 || true)"
    detail="total_tests=${total_tests:-n/a}"
    if grep -Eq "$EMPTY_TEST_RESULT_PATTERN" "$log_file"; then
      exit_code=1
      detail="$detail, empty-result"
    elif [[ -z "$total_tests" || "$total_tests" -lt "$MIN_INTEGRATION_TESTS" ]]; then
      exit_code=1
      detail="$detail, less-than-min-$MIN_INTEGRATION_TESTS"
    fi
  else
    detail="log-missing"
    exit_code=1
  fi

  append_summary "integration" "$([[ "$exit_code" == "0" ]] && echo success || echo failure)" "$detail"
  return "$exit_code"
}

job_fusion_body() {
  cd "$REPO_ROOT/test" || return $?
  timeout "$FUSION_TIMEOUT" "$PYTHON_BIN" test_runner.py --fusion --quiet
}

job_fusion() {
  local log_file="$LOG_ROOT/fusion/fusion_current.log"
  run_logged "fusion current" "$log_file" job_fusion_body
  local exit_code=$?
  append_summary "fusion" "$([[ "$exit_code" == "0" ]] && echo success || echo failure)" "log=$log_file"
  return "$exit_code"
}

job_t_perf_body() {
  cd "$MODEL_ROOT/inference/prunedGraph" || return $?
  timeout "$T_PERF_TIMEOUT" "$PYTHON_BIN" run_inference.py \
    --device musa \
    --batch-size 100 \
    --infer-iters 1000
}

parse_chinese_average_ms() {
  local log_file="$1"
  "$PYTHON_BIN" - "$log_file" <<'PY' || true
import pathlib
import re
import sys

text = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8", errors="ignore")
matches = re.findall(r"\u5e73\u5747:\s*([0-9]+(?:\.[0-9]+)?)\s*ms", text)
print(matches[-1] if matches else "")
PY
}

job_t_perf() {
  local log_file="$LOG_ROOT/t-perf/t_perf_current.log"
  run_logged "T performance current" "$log_file" run_with_optional_gpu_lock job_t_perf_body
  local exit_code=$?
  local current_ms=""

  if [[ "$exit_code" == "0" && -f "$log_file" ]]; then
    current_ms="$(parse_chinese_average_ms "$log_file")"
    [[ -n "$current_ms" ]] || exit_code=1
  fi

  append_summary "t_perf" "$([[ "$exit_code" == "0" ]] && echo success || echo failure)" "current_ms=${current_ms:-n/a}"
  return "$exit_code"
}

job_t_accuracy_body() {
  cd "$MODEL_ROOT/inference/prunedGraph" || return $?
  TF_ENABLE_ONEDNN_OPTS=0 \
  MUSA_ENABLE_TF32=0 \
  timeout "$T_ACCURACY_TIMEOUT" "$PYTHON_BIN" run_inference.py \
    --device musa \
    --batch-size 100 \
    --check-acc \
    --rtol 1e-2 \
    --atol 1e-2
}

job_t_accuracy() {
  local log_file="$LOG_ROOT/t-accuracy/t_accuracy_current.log"
  run_logged "T accuracy current" "$log_file" run_with_optional_gpu_lock job_t_accuracy_body
  local exit_code=$?
  local current_acc="n/a"

  if [[ -f "$log_file" ]]; then
    current_acc="$(grep -E "PASSED|FAILED" "$log_file" | tail -1 | grep -oE "PASSED|FAILED" || echo "n/a")"
  fi
  if [[ "$current_acc" != "PASSED" ]]; then
    exit_code=1
  fi

  append_summary "t_accuracy" "$([[ "$exit_code" == "0" ]] && echo success || echo failure)" "current=$current_acc"
  return "$exit_code"
}

bd_report_summary() {
  local log_file="$1"
  local report_path=""

  if [[ -f "$log_file" ]]; then
    report_path="$(grep '^\[OK\] report=' "$log_file" | tail -1 | sed 's/^\[OK\] report=//' || true)"
  fi

  if [[ -z "$report_path" || ! -f "$report_path" ]]; then
    printf 'report=n/a'
    return 1
  fi

  "$PYTHON_BIN" - "$report_path" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    report = json.load(f)

parts = []
ok = True
for item in report.get("average_time_summary", []):
    bs = item.get("batch_size")
    status = item.get("status")
    value = item.get("trimmed_avg_ms")
    if value is None:
        value = item.get("average_time_ms")
    if status != "ok" or value is None:
        ok = False
        parts.append(f"bs{bs}=n/a")
    else:
        parts.append(f"bs{bs}={float(value):.4f}ms")

print(", ".join(parts) if parts else "report-empty")
sys.exit(0 if ok and parts else 1)
PY
}

bd_model_body() {
  local model_id="$1"
  local job_dir="$2"
  local spec_path="$MODEL_ROOT/inference/metaGraph/meta_graph/meta_graph_${model_id}.spec"

  if [[ ! -f "$spec_path" ]]; then
    echo "Spec file not found: $spec_path"
    return 1
  fi

  cd "$MODEL_ROOT/inference/metaGraph" || return $?
  MUSA_PINNED_FEED=1 \
  MUSA_PINNED_H2D_ON_COMPUTE_STREAM=1 \
  timeout "$BD_MODEL_TIMEOUT" "$PYTHON_BIN" musa_run_pb_graph.py \
    --spec "$spec_path" \
    --bs "$BD_MODEL_BS" \
    --run_iters "$BD_MODEL_RUN_ITERS" \
    --out_root "$job_dir/current-out"
}

job_bd_model() {
  local model_id="$1"
  local job_name="bd_model${model_id}"
  local job_dir="$LOG_ROOT/bd-model${model_id}"
  local log_file="$job_dir/${job_name}_current.log"

  mkdir -p "$job_dir"
  rm -rf "$job_dir/current-out"

  run_logged "$job_name current" "$log_file" run_with_optional_gpu_lock bd_model_body "$model_id" "$job_dir"
  local exit_code=$?
  local detail=""
  detail="$(bd_report_summary "$log_file")" || exit_code=1

  append_summary "$job_name" "$([[ "$exit_code" == "0" ]] && echo success || echo failure)" "$detail"
  return "$exit_code"
}

find_first_npz() {
  local preferred_file="$1"
  local data_dir="$2"

  if [[ -n "$preferred_file" && -f "$preferred_file" ]]; then
    printf '%s\n' "$preferred_file"
    return 0
  fi

  if [[ -d "$data_dir" ]]; then
    find "$data_dir" -maxdepth 2 -type f -name '*.npz' | sort | head -n 1
  fi
}

training_perf_body() {
  local root="$1"
  local script_name="$2"
  local data_dir="$3"
  local explicit_data_file="$4"
  local warmup_steps="$5"
  local measure_steps="$6"
  local timeout_value="$7"

  local script_path="$root/$script_name"
  local data_file

  if [[ ! -d "$root" ]]; then
    echo "Training performance root not found: $root"
    return 1
  fi
  if [[ ! -d "$root/model" || ! -d "$root/data" ]]; then
    echo "Training performance model/data directories not found under: $root"
    return 1
  fi
  if [[ ! -f "$script_path" ]]; then
    echo "Training performance script missing: $script_path"
    return 1
  fi

  data_file="$(find_first_npz "$explicit_data_file" "$data_dir")"
  if [[ -z "$data_file" || ! -f "$data_file" ]]; then
    echo "Training performance data npz not found under: $data_dir"
    return 1
  fi

  cd "$root" || return $?
  echo "Script: $script_path"
  echo "Data file: $data_file"
  timeout "$timeout_value" "$PYTHON_BIN" -u "$script_path" \
    --data_path "$data_file" \
    --warmup_steps "$warmup_steps" \
    --measure_steps "$measure_steps" \
    --max_steps "$((warmup_steps + measure_steps))"
}

parse_average_full_step_ms() {
  local log_file="$1"
  "$PYTHON_BIN" - "$log_file" <<'PY' || true
import pathlib
import re
import sys

text = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8", errors="ignore")
matches = re.findall(r"Average ms/full step:\s*([0-9]+(?:\.[0-9]+)?)", text)
print(matches[-1] if matches else "")
PY
}

job_onetrans_perf() {
  local log_file="$LOG_ROOT/onetrans-perf/onetrans_perf_current.log"
  run_logged "OneTrans performance current" "$log_file" run_with_optional_gpu_lock training_perf_body \
    "$ONETRANS_PERF_ROOT" \
    "$ONETRANS_PERF_SCRIPT_NAME" \
    "$ONETRANS_PERF_DATA_DIR" \
    "${ONETRANS_PERF_DATA_FILE:-}" \
    "$ONETRANS_PERF_WARMUP_STEPS" \
    "$ONETRANS_PERF_MEASURE_STEPS" \
    "$ONETRANS_PERF_TIMEOUT"
  local exit_code=$?
  local current_ms=""

  if [[ "$exit_code" == "0" && -f "$log_file" ]]; then
    current_ms="$(parse_average_full_step_ms "$log_file")"
    [[ -n "$current_ms" ]] || exit_code=1
  fi

  append_summary "onetrans_perf" "$([[ "$exit_code" == "0" ]] && echo success || echo failure)" "current_ms=${current_ms:-n/a}"
  return "$exit_code"
}

job_tokenlarge_perf() {
  local log_file="$LOG_ROOT/tokenlarge-perf/tokenlarge_perf_current.log"
  run_logged "TokenMixer Large performance current" "$log_file" run_with_optional_gpu_lock training_perf_body \
    "$TOKENLARGE_PERF_ROOT" \
    "$TOKENLARGE_PERF_SCRIPT_NAME" \
    "$TOKENLARGE_PERF_DATA_DIR" \
    "${TOKENLARGE_PERF_DATA_FILE:-}" \
    "$TOKENLARGE_PERF_WARMUP_STEPS" \
    "$TOKENLARGE_PERF_MEASURE_STEPS" \
    "$TOKENLARGE_PERF_TIMEOUT"
  local exit_code=$?
  local current_ms=""

  if [[ "$exit_code" == "0" && -f "$log_file" ]]; then
    current_ms="$(parse_average_full_step_ms "$log_file")"
    [[ -n "$current_ms" ]] || exit_code=1
  fi

  append_summary "tokenlarge_perf" "$([[ "$exit_code" == "0" ]] && echo success || echo failure)" "current_ms=${current_ms:-n/a}"
  return "$exit_code"
}

job_training_body() {
  local job_dir="$LOG_ROOT/training"
  local error_log_dir="$job_dir/error_logs"
  local live_wrapper="$job_dir/run_training_live.py"
  local selected_models="$TRAINING_MODELS"

  if [[ -z "$selected_models" ]]; then
    if [[ "$TRAINING_MODE" == "daily" ]]; then
      selected_models="$DAILY_TRAINING_MODELS"
    else
      selected_models="$PR_TRAINING_MODELS"
    fi
  fi

  if [[ ! -d "$MODEL_ROOT/training" ]]; then
    echo "Training root not found: $MODEL_ROOT/training"
    return 1
  fi

  mkdir -p "$job_dir" "$error_log_dir"
  cat > "$live_wrapper" <<'PY'
import os
import selectors
import subprocess
import sys
import time

sys.path.insert(0, os.getcwd())
import run_all_training_tests as runner

_real_run = subprocess.run


def run_live(cmd, capture_output=False, text=False, cwd=None, env=None, timeout=None, **kwargs):
    if not capture_output:
        return _real_run(cmd, capture_output=capture_output, text=text, cwd=cwd, env=env, timeout=timeout, **kwargs)

    effective_timeout = int(env.get("TRAINING_MODEL_TIMEOUT_SECONDS", "3600")) if env else 3600
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=cwd,
        env=env,
    )

    lines = []
    start = time.monotonic()
    selector = selectors.DefaultSelector()
    selector.register(proc.stdout, selectors.EVENT_READ)

    try:
        while True:
            for key, _ in selector.select(timeout=0.2):
                line = key.fileobj.readline()
                if line:
                    lines.append(line)
                    sys.stdout.write(line)
                    sys.stdout.flush()

            if proc.poll() is not None:
                rest = proc.stdout.read()
                if rest:
                    lines.append(rest)
                    sys.stdout.write(rest)
                    sys.stdout.flush()
                break

            if effective_timeout > 0 and time.monotonic() - start > effective_timeout:
                proc.kill()
                proc.wait()
                timeout_msg = f"\n[ERROR] Model command timeout (>{effective_timeout}s)\n"
                lines.append(timeout_msg)
                sys.stdout.write(timeout_msg)
                sys.stdout.flush()
                return subprocess.CompletedProcess(cmd, 124, stdout="".join(lines), stderr="")
    finally:
        selector.close()
        if proc.stdout:
            proc.stdout.close()

    return subprocess.CompletedProcess(cmd, proc.returncode, stdout="".join(lines), stderr="")


runner.subprocess.run = run_live
sys.exit(runner.main())
PY

  cd "$MODEL_ROOT/training" || return $?
  read -r -a training_models <<< "$selected_models"
  echo "Training mode: $TRAINING_MODE"
  echo "Selected models (${#training_models[@]}): $selected_models"

  TRAINING_MODEL_TIMEOUT_SECONDS="$TRAINING_MODEL_TIMEOUT_SECONDS" \
  PYTHONUNBUFFERED=1 \
  timeout "$TRAINING_TIMEOUT" "$PYTHON_BIN" -u "$live_wrapper" \
    --epochs "$TRAINING_EPOCHS" \
    --log-dir "$error_log_dir" \
    --models "${training_models[@]}"
}

job_training() {
  local log_file="$LOG_ROOT/training/training_tests_current.log"
  run_logged "training current" "$log_file" run_with_optional_gpu_lock job_training_body
  local exit_code=$?
  local detail="log=$log_file"

  if [[ ! -f "$log_file" ]]; then
    exit_code=1
    detail="log-missing"
  elif grep -Eq "$EMPTY_TEST_RESULT_PATTERN" "$log_file" || grep -q '^\[FAIL\]' "$log_file"; then
    exit_code=1
    detail="failure-marker"
  fi

  append_summary "training" "$([[ "$exit_code" == "0" ]] && echo success || echo failure)" "$detail"
  return "$exit_code"
}

write_summary_file() {
  local summary_file="$LOG_ROOT/summaries/current-only-summary.md"
  {
    echo "## Current-only Validation"
    echo
    echo "- Repo: $REPO_ROOT"
    echo "- Log root: $LOG_ROOT"
    echo "- Model root: $MODEL_ROOT"
    echo "- Jobs: $CURRENT_ONLY_JOBS"
    echo "- Wheel: ${RESOLVED_WHEEL_PATH:-n/a}"
    echo "- GPU lock: ${GPU_LOCK_FILE:-not set}"
    echo
    echo "| Job | Status | Detail |"
    echo "|-----|--------|--------|"
    local row
    for row in "${SUMMARY_ROWS[@]}"; do
      IFS='|' read -r name status detail <<< "$row"
      echo "| $name | $status | $detail |"
    done
  } | tee "$summary_file"
}

run_job_by_name() {
  local job="$1"
  case "$job" in
    integration)
      job_integration
      ;;
    fusion)
      job_fusion
      ;;
    t_accuracy)
      job_t_accuracy
      ;;
    t_perf)
      job_t_perf
      ;;
    bd_model1)
      job_bd_model 1
      ;;
    bd_model2)
      job_bd_model 2
      ;;
    bd_model3)
      job_bd_model 3
      ;;
    onetrans_perf)
      job_onetrans_perf
      ;;
    tokenlarge_perf)
      job_tokenlarge_perf
      ;;
    training)
      job_training
      ;;
    *)
      log_error "Unknown job: $job"
      return 1
      ;;
  esac
}

main() {
  local any_failed=0

  if [[ "$CURRENT_ONLY_JOBS" == "all" ]]; then
    CURRENT_ONLY_JOBS="$ALL_CURRENT_ONLY_JOBS"
  fi

  RESOLVED_WHEEL_PATH="$(resolve_wheel_path)" || exit 1
  log_info "Repo: $REPO_ROOT"
  log_info "Log root: $LOG_ROOT"
  log_info "Wheel: $RESOLVED_WHEEL_PATH"
  log_info "Jobs: $CURRENT_ONLY_JOBS"

  install_current_wheel "$RESOLVED_WHEEL_PATH" || exit 1

  local job
  for job in $CURRENT_ONLY_JOBS; do
    if run_job_by_name "$job"; then
      :
    else
      any_failed=1
      if [[ "$FAIL_FAST" == "1" ]]; then
        break
      fi
    fi
  done

  write_summary_file

  if [[ "$any_failed" == "0" ]]; then
    log_info "Current-only validation finished successfully."
  else
    log_error "Current-only validation finished with failures."
  fi
  return "$any_failed"
}

main "$@"
