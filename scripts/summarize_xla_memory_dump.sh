#!/usr/bin/env bash
set -euo pipefail

dump_dir="${1:-./xla_dump_mem}"
out_file="${2:-xla_memory_summary.md}"

if [[ ! -d "${dump_dir}" ]]; then
    echo "XLA dump directory not found: ${dump_dir}" >&2
    exit 1
fi

hlo_list="$(mktemp)"
buffer_list="$(mktemp)"
trap 'rm -f "${hlo_list}" "${buffer_list}"' EXIT

find "${dump_dir}" -name "*cpu_after_optimizations.txt" | sort > "${hlo_list}"
find "${dump_dir}" -name "*cpu_after_optimizations-buffer-assignment.txt" | sort > "${buffer_list}"

hlo_count="$(wc -l < "${hlo_list}" | tr -d ' ')"
buffer_count="$(wc -l < "${buffer_list}" | tr -d ' ')"

{
    echo "# XLA Memory Dump Summary"
    echo
    echo "- dump_dir: \`${dump_dir}\`"
    echo "- hlo_modules: ${hlo_count}"
    echo "- buffer_assignment_modules: ${buffer_count}"
    echo

    echo "## Module Index"
    echo
    while IFS= read -r hlo; do
        module="$(basename "${hlo}" .txt)"
        source_hint="$(grep -m1 -E "pure_jax.py|ffi_cpu.py|operator.py" "${hlo}" || true)"
        entry="$(grep -m1 -E "^ENTRY " "${hlo}" || true)"
        custom_call="$(grep -m1 -E "custom_call_target=|custom-call" "${hlo}" || true)"
        if grep -q "pure_jax.py" "${hlo}"; then
            kind="pure_jax"
        elif grep -q "ffi_cpu.py" "${hlo}"; then
            kind="ffi_cpu"
        else
            kind="unknown"
        fi
        echo "### ${module}"
        echo
        echo "- inferred_kind: ${kind}"
        echo "- source_hint: \`${source_hint}\`"
        echo "- entry: \`${entry}\`"
        echo "- custom_call_hint: \`${custom_call}\`"
        echo
    done < "${hlo_list}"

    echo "## Per-Module Operation Counts"
    echo
    echo "| module | kind | custom-call | while | scatter | dynamic-update-slice | dynamic-slice | copy | f32[*,128] refs |"
    echo "|---|---|---:|---:|---:|---:|---:|---:|---:|"
    while IFS= read -r hlo; do
        module="$(basename "${hlo}" .txt)"
        if grep -q "pure_jax.py" "${hlo}"; then
            kind="pure_jax"
        elif grep -q "ffi_cpu.py" "${hlo}"; then
            kind="ffi_cpu"
        else
            kind="unknown"
        fi
        custom_calls="$(grep -c "custom-call" "${hlo}" || true)"
        whiles="$(grep -c " while(" "${hlo}" || true)"
        scatters="$(grep -c "scatter" "${hlo}" || true)"
        dus="$(grep -c "dynamic-update-slice" "${hlo}" || true)"
        ds="$(grep -c "dynamic-slice" "${hlo}" || true)"
        copies="$(grep -c "copy" "${hlo}" || true)"
        f32_128="$(grep -c -E "f32\\[[0-9,]+128\\]" "${hlo}" || true)"
        echo "| ${module} | ${kind} | ${custom_calls} | ${whiles} | ${scatters} | ${dus} | ${ds} | ${copies} | ${f32_128} |"
    done < "${hlo_list}"
    echo

    echo "## Buffer Assignment Totals"
    echo
    echo "| module | total bytes line |"
    echo "|---|---|"
    while IFS= read -r buf; do
        module="$(basename "${buf}" .txt)"
        total="$(grep -m1 "Total bytes used" "${buf}" || true)"
        echo "| ${module} | \`${total}\` |"
    done < "${buffer_list}"
    echo

    echo "## Largest Allocations"
    echo
    while IFS= read -r buf; do
        module="$(basename "${buf}" .txt)"
        echo
        echo "### ${module}"
        echo
        echo '```text'
        grep -nE "allocation [0-9]+:|size " "${buf}" | head -120 || true
        echo '```'
    done < "${buffer_list}"
    echo

    echo "## Alias / Live-Out Evidence"
    echo
    while IFS= read -r buf; do
        module="$(basename "${buf}" .txt)"
        echo
        echo "### ${module}"
        echo
        echo '```text'
        grep -nE "alias|Alias|must-alias|may-alias|live-out|maybe-live-out" "${buf}" | head -120 || true
        echo '```'
    done < "${buffer_list}"
    echo

    echo "## FFI Custom Calls"
    echo
    while IFS= read -r hlo; do
        module="$(basename "${hlo}" .txt)"
        echo
        echo "### ${module}"
        echo
        echo '```text'
        grep -nE "custom-call|custom_call_target|ffi_call|linear_dag" "${hlo}" | head -120 || true
        echo '```'
    done < "${hlo_list}"
    echo

    echo "## Explicit Copy Ops"
    echo
    while IFS= read -r hlo; do
        module="$(basename "${hlo}" .txt)"
        echo
        echo "### ${module}"
        echo
        echo '```text'
        grep -nE " copy\\(| copy\\.|copy =" "${hlo}" | head -120 || true
        echo '```'
    done < "${hlo_list}"
    echo

    echo "## Large State References"
    echo
    while IFS= read -r hlo; do
        module="$(basename "${hlo}" .txt)"
        echo
        echo "### ${module}"
        echo
        echo '```text'
        grep -nE "f32\\[[0-9]+,128\\]" "${hlo}" | head -160 || true
        echo '```'
    done < "${hlo_list}"
} > "${out_file}"

echo "Wrote ${out_file}"
