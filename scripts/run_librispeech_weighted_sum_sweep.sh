#!/usr/bin/env bash
set -euo pipefail

# Run the complete sweep sequentially on one machine/GPU. Supplying task IDs
# runs only those jobs, for example:
#
#   scripts/run_librispeech_weighted_sum_sweep.sh 0 28
#
# Completed jobs have a .complete marker in their output directory and are
# skipped when this script is restarted. Set FORCE=1 to rerun them.

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="${PROJECT_ROOT:-$(cd -- "${script_dir}/.." && pwd)}"
task_script="${script_dir}/run_librispeech_weighted_sum_task.sh"
log_root="${LOG_ROOT:-${project_root}/logs/librispeech_weighted_sum}"

if (( $# > 0 )); then
    task_ids=("$@")
else
    task_ids=({0..55})
fi

mkdir -p "${log_root}"
failed=()

for task_id in "${task_ids[@]}"; do
    printf -v padded_task_id '%02d' "${task_id}"
    log_path="${log_root}/task_${padded_task_id}.log"

    if "${task_script}" "${task_id}" 2>&1 | tee "${log_path}"; then
        continue
    fi

    failed+=("${task_id}")
    if [[ "${CONTINUE_ON_ERROR:-0}" != "1" ]]; then
        echo "task ${task_id} failed; stopping the sweep" >&2
        echo "rerun this script to retry it and continue" >&2
        exit 1
    fi
done

if (( ${#failed[@]} > 0 )); then
    printf 'failed task IDs:' >&2
    printf ' %s' "${failed[@]}" >&2
    printf '\n' >&2
    exit 1
fi

echo "all requested sweep tasks completed"

