#!/usr/bin/env bash
# pattern: Imperative Shell
# This runner owns cross-machine build, validation, cache, logging, and artifact
# orchestration. Promotion matching and decisions remain in the Python core.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: run_jax_promotion.sh \
  --repo-root ABSOLUTE_PATH \
  --hdf5-path ABSOLUTE_PATH \
  --output-dir ABSOLUTE_PATH \
  --platform-label arm64-cpu|x86_64-cpu|forced-two-device-cpu|gpu \
  --device-count N \
  [--no-enforce-gates] [--allow-dirty] [--dry-run]

The output directory must already exist. Promotable runs require a clean
candidate checkout and an output directory outside that checkout. --allow-dirty
is only valid with --no-enforce-gates and produces non-promotable evidence.
EOF
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 2
}

resolve_existing_path() {
    local path=$1
    [[ -e "$path" ]] || die "path does not exist: $path"
    if command -v realpath >/dev/null 2>&1; then
        realpath "$path"
        return
    fi
    local directory
    directory=$(cd "$(dirname "$path")" && pwd -P)
    printf '%s/%s\n' "$directory" "$(basename "$path")"
}

repo_root_arg=''
hdf5_path_arg=''
output_dir_arg=''
platform_label=''
device_count=''
enforce_gates=true
allow_dirty=false
dry_run=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo-root)
            [[ $# -ge 2 ]] || die "--repo-root requires a value"
            repo_root_arg=$2
            shift 2
            ;;
        --hdf5-path)
            [[ $# -ge 2 ]] || die "--hdf5-path requires a value"
            hdf5_path_arg=$2
            shift 2
            ;;
        --output-dir)
            [[ $# -ge 2 ]] || die "--output-dir requires a value"
            output_dir_arg=$2
            shift 2
            ;;
        --platform-label)
            [[ $# -ge 2 ]] || die "--platform-label requires a value"
            platform_label=$2
            shift 2
            ;;
        --device-count)
            [[ $# -ge 2 ]] || die "--device-count requires a value"
            device_count=$2
            shift 2
            ;;
        --no-enforce-gates)
            enforce_gates=false
            shift
            ;;
        --allow-dirty)
            allow_dirty=true
            shift
            ;;
        --dry-run)
            dry_run=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown argument: $1"
            ;;
    esac
done

[[ -n "$repo_root_arg" ]] || die "--repo-root is required"
[[ -n "$hdf5_path_arg" ]] || die "--hdf5-path is required"
[[ -n "$output_dir_arg" ]] || die "--output-dir is required"
[[ -n "$platform_label" ]] || die "--platform-label is required"
[[ -n "$device_count" ]] || die "--device-count is required"
[[ "$device_count" =~ ^[1-9][0-9]*$ ]] || die "--device-count must be a positive integer"

case "$platform_label" in
    arm64-cpu|x86_64-cpu)
        [[ "$device_count" -eq 1 ]] || die "$platform_label requires --device-count 1"
        device_platform='cpu'
        backend_mode='cpu_ffi_and_pure_jax'
        accelerator_backend='ffi_cpu'
        ;;
    forced-two-device-cpu)
        [[ "$device_count" -eq 2 ]] || die "forced-two-device-cpu requires --device-count 2"
        device_platform='cpu'
        backend_mode='cpu_ffi_and_pure_jax'
        accelerator_backend='ffi_cpu'
        ;;
    gpu)
        device_platform='gpu'
        backend_mode='pure_jax'
        accelerator_backend='none'
        ;;
    *)
        die "unsupported --platform-label: $platform_label"
        ;;
esac

repo_root=$(resolve_existing_path "$repo_root_arg")
hdf5_path=$(resolve_existing_path "$hdf5_path_arg")
output_dir=$(resolve_existing_path "$output_dir_arg")
[[ -d "$repo_root" ]] || die "--repo-root must name a directory"
[[ -f "$hdf5_path" ]] || die "--hdf5-path must name a file"
[[ -d "$output_dir" ]] || die "--output-dir must name an existing directory"

actual_repo_root=$(git -C "$repo_root" rev-parse --show-toplevel 2>/dev/null) || die "--repo-root is not a Git checkout"
actual_repo_root=$(resolve_existing_path "$actual_repo_root")
[[ "$actual_repo_root" == "$repo_root" ]] || die "--repo-root must be the checkout root: $actual_repo_root"

case "$repo_root/" in
    "$output_dir/"*) die "unsafe output directory: it is the repository root or an ancestor" ;;
esac
if [[ "$output_dir" == "$repo_root" || "$output_dir" == "/" ]]; then
    die "unsafe output directory: $output_dir"
fi

output_inside_repo=false
case "$output_dir/" in
    "$repo_root/"*) output_inside_repo=true ;;
esac
if [[ "$output_inside_repo" == true && "$allow_dirty" == false ]]; then
    die "promotable output directory must be outside the repository checkout"
fi

git_status=$(git -C "$repo_root" status --porcelain --untracked-files=all)
relevant_status=''
while IFS= read -r status_line; do
    [[ -n "$status_line" ]] || continue
    status_path=${status_line:3}
    status_path=${status_path#\"}
    status_path=${status_path%\"}
    if [[ "$status_line" == '?? '* ]]; then
        case "$status_path" in
            *.h5|*.H5|*.hdf5|*.HDF5) continue ;;
        esac
    fi
    relevant_status+="$status_line"$'\n'
done <<< "$git_status"

candidate_clean=true
if [[ -n "$relevant_status" ]]; then
    candidate_clean=false
fi
if [[ "$candidate_clean" == false && "$allow_dirty" == false ]]; then
    die "promotable evidence requires a clean candidate commit"
fi
if [[ "$allow_dirty" == true && "$enforce_gates" == true ]]; then
    die "--allow-dirty requires --no-enforce-gates"
fi

candidate_commit=$(git -C "$repo_root" rev-parse HEAD)
promotable=true
if [[ "$candidate_clean" == false || "$enforce_gates" == false || "$dry_run" == true ]]; then
    promotable=false
fi

for state in fresh reused; do
    for suffix in evidence.json commands.log environment.log execution.log; do
        artifact="$output_dir/$platform_label.$state.$suffix"
        [[ ! -e "$artifact" ]] || die "refusing to overwrite existing artifact: $artifact"
    done
done
for suffix in setup.commands.log setup.environment.log setup.execution.log; do
    artifact="$output_dir/$platform_label.$suffix"
    [[ ! -e "$artifact" ]] || die "refusing to overwrite existing artifact: $artifact"
done
[[ ! -e "$output_dir/checksums.sha256" ]] || \
    die "refusing to overwrite existing artifact: $output_dir/checksums.sha256"

runner_dir=$(mktemp -d "$output_dir/.jax-promotion-run.XXXXXX")
runner_dir=$(resolve_existing_path "$runner_dir")
case "$runner_dir/" in
    "$output_dir/.jax-promotion-run."*) ;;
    *) die "runner temporary directory escaped output directory" ;;
esac

benchmark_cache="$runner_dir/shared-persistent-cache"
validation_cache="$runner_dir/validation-cache"
mkdir -p "$benchmark_cache" "$validation_cache"
if [[ -n "$(find "$benchmark_cache" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    die "fresh benchmark cache is unexpectedly non-empty"
fi

secret_values=('')
while IFS='=' read -r environment_name _; do
    case "$environment_name" in
        *TOKEN*|*PASSWORD*|*SECRET*|*CREDENTIAL*|*AUTH*|*_KEY|*_KEY_*|KEY_*)
            environment_value=${!environment_name}
            if [[ ${#environment_value} -ge 4 ]]; then
                secret_values+=("$environment_value")
            fi
            ;;
    esac
done < <(env)

redact_text() {
    local value=$1
    value=${value//"$hdf5_path"/<dataset>}
    value=${value//"$runner_dir"/<runner-temp>}
    value=${value//"$output_dir"/<output>}
    value=${value//"$repo_root"/<repo>}
    if [[ -n "${HOME:-}" ]]; then
        value=${value//"$HOME"/<home>}
    fi
    if [[ -n "${USER:-}" ]]; then
        value=${value//"$USER"/<user>}
    fi
    if [[ -n "${HOSTNAME:-}" ]]; then
        value=${value//"$HOSTNAME"/<host>}
    fi
    local secret_value
    for secret_value in "${secret_values[@]}"; do
        if [[ -n "$secret_value" ]]; then
            value=${value//"$secret_value"/<redacted-secret>}
        fi
    done
    printf '%s' "$value"
}

redact_file() {
    local source=$1
    local destination=$2
    while IFS= read -r line || [[ -n "$line" ]]; do
        redact_text "$line" >> "$destination"
        printf '\n' >> "$destination"
    done < "$source"
}

append_command() {
    local destination=$1
    shift
    local rendered=''
    local argument
    local quoted
    for argument in "$@"; do
        printf -v quoted '%q' "$argument"
        rendered+="$quoted "
    done
    rendered=${rendered% }
    printf 'command=%s\n' "$(redact_text "$rendered")" >> "$destination"
}

run_logged() {
    local label=$1
    local command_log=$2
    local execution_log=$3
    shift 3
    append_command "$command_log" "$@"
    if [[ "$dry_run" == true ]]; then
        printf 'dry-run: %s was not executed\n' "$label" >> "$execution_log"
        return
    fi
    local raw_log="$runner_dir/$label.raw.log"
    local status
    set +e
    (cd "$repo_root" && "$@") > "$raw_log" 2>&1
    status=$?
    set -e
    redact_file "$raw_log" "$execution_log"
    if [[ "$status" -ne 0 ]]; then
        printf 'error: %s failed with exit code %s; runner state retained at %s\n' \
            "$label" "$status" "$(redact_text "$runner_dir")" >&2
        exit "$status"
    fi
}

write_environment_log() {
    local destination=$1
    local cache_policy=$2
    local process_id=$3
    local cache_directory_id=$4
    {
        printf 'candidate_commit=%s\n' "$candidate_commit"
        printf 'candidate_clean=%s\n' "$candidate_clean"
        printf 'promotable=%s\n' "$promotable"
        printf 'repository_path=<repo>\n'
        printf 'dataset_path=<dataset>\n'
        printf 'output_path=<output>\n'
        printf 'platform_label=%s\n' "$platform_label"
        printf 'architecture=%s\n' "$(uname -m)"
        printf 'device_count=%s\n' "$device_count"
        printf 'device_platform=%s\n' "$device_platform"
        printf 'backend_mode=%s\n' "$backend_mode"
        printf 'accelerator_specific_backend=%s\n' "$accelerator_backend"
        printf 'cache_policy=%s\n' "$cache_policy"
        printf 'cache_directory_id=%s\n' "$cache_directory_id"
        printf 'process_id=%s\n' "$process_id"
        printf 'promotion_gate_enforcement=%s\n' "$enforce_gates"
    } > "$destination"
}

setup_commands="$output_dir/$platform_label.setup.commands.log"
setup_environment="$output_dir/$platform_label.setup.environment.log"
setup_execution="$output_dir/$platform_label.setup.execution.log"
: > "$setup_commands"
: > "$setup_execution"
write_environment_log "$setup_environment" 'validation' 'setup-validation' 'validation-only-cache'

runtime_env=(
    "LINEAR_DAG_PROMOTION_DEVICE_PLATFORM=$device_platform"
    "JAX_COMPILATION_CACHE_DIR=$validation_cache"
    "XLA_CACHE_DIR=$validation_cache"
    "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0"
    "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1"
)
if [[ "$device_platform" == 'cpu' ]]; then
    runtime_env+=("JAX_PLATFORMS=cpu" "XLA_FLAGS=--xla_force_host_platform_device_count=$device_count")
    build_command=(env LINEAR_DAG_REQUIRE_FFI_CPU=1 uv build)
    correctness_tests=(
        tests/jax/test_packing.py
        tests/jax/test_packed_products.py
        tests/jax/test_hijax.py
        tests/jax/test_transform_composition.py
        tests/jax/test_grm_operator.py
        tests/jax/test_operator_ffi_cpu.py
        tests/jax/test_kernels_ffi_cpu.py
        tests/association/test_heritability_jax.py
    )
else
    runtime_env+=("JAX_PLATFORMS=gpu")
    build_command=(uv build)
    correctness_tests=(
        tests/jax/test_packing.py
        tests/jax/test_packed_products.py
        tests/jax/test_hijax.py
        tests/jax/test_transform_composition.py
        tests/jax/test_grm_operator.py
        tests/association/test_heritability_jax.py
    )
fi

correctness_command=(uv run pytest -p no:capture "${correctness_tests[@]}")
if [[ "$device_platform" == 'gpu' ]]; then
    correctness_command+=(-k 'not ffi')
fi

run_logged 'build' "$setup_commands" "$setup_execution" "${build_command[@]}"
run_logged \
    'correctness-float32' \
    "$setup_commands" \
    "$setup_execution" \
    env "${runtime_env[@]}" JAX_ENABLE_X64=0 \
    "${correctness_command[@]}"
run_logged \
    'correctness-float64' \
    "$setup_commands" \
    "$setup_execution" \
    env "${runtime_env[@]}" JAX_ENABLE_X64=1 \
    "${correctness_command[@]}"

if command -v sha256sum >/dev/null 2>&1; then
    validation_digest=$(sha256sum "$setup_commands" "$setup_execution" | sha256sum | awk '{print $1}')
else
    validation_digest=$(shasum -a 256 "$setup_commands" "$setup_execution" | shasum -a 256 | awk '{print $1}')
fi
validation_evidence_id="setup-logs-sha256:$validation_digest"

for cache_policy in fresh reused; do
    evidence_path="$output_dir/$platform_label.$cache_policy.evidence.json"
    command_log="$output_dir/$platform_label.$cache_policy.commands.log"
    environment_log="$output_dir/$platform_label.$cache_policy.environment.log"
    execution_log="$output_dir/$platform_label.$cache_policy.execution.log"
    : > "$command_log"
    : > "$execution_log"
    write_environment_log \
        "$environment_log" \
        "$cache_policy" \
        "benchmark-$cache_policy" \
        'shared-persistent-cache'

    benchmark_env=(
        "LINEAR_DAG_PROMOTION_DEVICE_PLATFORM=$device_platform"
        "JAX_COMPILATION_CACHE_DIR=$benchmark_cache"
        "XLA_CACHE_DIR=$benchmark_cache"
        "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0"
        "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1"
        "JAX_ENABLE_X64=1"
    )
    if [[ "$device_platform" == 'cpu' ]]; then
        benchmark_env+=("JAX_PLATFORMS=cpu" "XLA_FLAGS=--xla_force_host_platform_device_count=$device_count")
    else
        benchmark_env+=("JAX_PLATFORMS=gpu")
    fi

    benchmark_command=(
        env "${benchmark_env[@]}"
        uv run pytest -p no:capture
        tests/jax/bench/test_promotion_benchmarks.py
        --runbench
        --jax-promotion-output "$evidence_path"
        --linarg-h5-path "$hdf5_path"
        --linarg-parallel-processes "$device_count"
        --linarg-benchmark-k 4 20
        --rhe-benchmark-num-matvecs 4 20
        --cache-policy "$cache_policy"
        --platform-label "$platform_label"
        --jax-validation-evidence-id "$validation_evidence_id"
    )
    if [[ "$enforce_gates" == true ]]; then
        benchmark_command+=(--jax-enforce-promotion-gates)
    fi
    run_logged "benchmark-$cache_policy" "$command_log" "$execution_log" "${benchmark_command[@]}"
    if [[ "$dry_run" == false && ! -s "$evidence_path" ]]; then
        die "benchmark did not write evidence: $evidence_path"
    fi
done

checksum_file="$output_dir/checksums.sha256"
checksum_temp="$runner_dir/checksums.sha256"
: > "$checksum_temp"
for artifact in "$output_dir/$platform_label".*.log "$output_dir/$platform_label".*.evidence.json; do
    [[ -f "$artifact" ]] || continue
    if command -v sha256sum >/dev/null 2>&1; then
        digest=$(sha256sum "$artifact" | awk '{print $1}')
    else
        digest=$(shasum -a 256 "$artifact" | awk '{print $1}')
    fi
    printf '%s  %s\n' "$digest" "$(basename "$artifact")" >> "$checksum_temp"
done
mv "$checksum_temp" "$checksum_file"

case "$runner_dir/" in
    "$output_dir/.jax-promotion-run."*) rm -rf -- "$runner_dir" ;;
    *) die "refusing to clean unverified runner directory" ;;
esac

printf 'promotion runner completed for %s (%s cache states); artifacts: %s\n' \
    "$platform_label" 2 "$output_dir"
