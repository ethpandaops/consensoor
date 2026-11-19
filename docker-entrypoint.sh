#!/bin/bash
set -eu

if [[ -z "${CONSENSOOR_LEANVM_SOCKET:-}" && "${LEANVM_SIDECAR:-1}" != "0" ]] && command -v leanvm-sidecar >/dev/null; then
    socket_dir=$(mktemp -d "${TMPDIR:-/tmp}/leanvm.XXXXXXXX")
    export CONSENSOOR_LEANVM_SOCKET="$socket_dir/prover.sock"
    leanvm-sidecar --socket "$CONSENSOOR_LEANVM_SOCKET" --threads "${LEANVM_THREADS:-0}" &
    prover_pid=$!
    deadline=$((SECONDS + 120))
    until [[ -S "$CONSENSOOR_LEANVM_SOCKET" ]]; do
        if ! kill -0 "$prover_pid" 2>/dev/null; then
            echo 'leanvm-sidecar exited before creating its socket' >&2
            exit 1
        fi
        if (( SECONDS >= deadline )); then
            echo 'leanvm-sidecar did not create its socket within 120 seconds' >&2
            exit 1
        fi
        sleep 0.1
    done
fi

exec consensoor "$@"
