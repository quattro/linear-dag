#!/usr/bin/env bash
set -euo pipefail

json_output=""
if [[ "${1:-}" == "--json-output" ]]; then
    if [[ $# -lt 2 ]]; then
        echo "usage: $0 --json-output PATH [DUMP_DIR]" >&2
        exit 2
    fi
    json_output="$2"
    shift 2
fi

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

if [[ -n "${json_output}" ]]; then
    python3 - "${dump_dir}" "${json_output}" <<'PY'
import json
import re
import sys
from pathlib import Path

dump_dir = Path(sys.argv[1])
output_path = Path(sys.argv[2])
hlo_suffix = ".cpu_after_optimizations.txt"
buffer_suffix = ".cpu_after_optimizations-buffer-assignment.txt"
hlo_paths = sorted(dump_dir.glob(f"*{hlo_suffix}"))
buffer_paths = sorted(dump_dir.glob(f"*{buffer_suffix}"))
modules = {}

for path in hlo_paths:
    name = path.name.removesuffix(hlo_suffix)
    text = path.read_text(encoding="utf-8", errors="replace")
    custom_calls = [line.strip() for line in text.splitlines() if "custom-call" in line or "custom_call_target" in line]
    modules.setdefault(name, {"module": name, "buffer_assignment_bytes": 0, "large_allocations": [], "aliases": [], "custom_calls": []})
    modules[name]["custom_calls"] = custom_calls

total_bytes = 0
for path in buffer_paths:
    name = path.name.removesuffix(buffer_suffix)
    text = path.read_text(encoding="utf-8", errors="replace")
    total_match = re.search(r"Total bytes used:\s*([0-9,]+)", text)
    module_bytes = int(total_match.group(1).replace(",", "")) if total_match else 0
    allocations = [int(value.replace(",", "")) for value in re.findall(r"allocation\s+\d+:\s+size\s+([0-9,]+)", text)]
    aliases = [line.strip() for line in text.splitlines() if re.search(r"alias|live-out", line, re.IGNORECASE)]
    module = modules.setdefault(name, {"module": name, "buffer_assignment_bytes": 0, "large_allocations": [], "aliases": [], "custom_calls": []})
    module["buffer_assignment_bytes"] = module_bytes
    module["large_allocations"] = sorted(allocations, reverse=True)
    module["aliases"] = aliases
    total_bytes += module_bytes

normalized = []
for module in sorted(modules.values(), key=lambda item: item["module"]):
    module["alias_count"] = sum(len(re.findall(r"alias|live-out", line, re.IGNORECASE)) for line in module["aliases"])
    module["custom_call_count"] = len(module["custom_calls"])
    normalized.append(module)

payload = {
    "schema_version": "2026-08-13+1",
    "hlo_module_count": len(hlo_paths),
    "buffer_assignment_module_count": len(buffer_paths),
    "total_buffer_assignment_bytes": total_bytes,
    "modules": normalized,
}
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
PY
    echo "Wrote ${json_output}"
    exit 0
fi

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
