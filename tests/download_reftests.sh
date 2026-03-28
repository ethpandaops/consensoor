#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <spec_tests_dir> [spec_version]"
  echo "  spec_version: 'nightly', 'nightly-YYYY-MM-DD', 'nightly-<run_id>',"
  echo "                or a release tag (e.g. v1.5.0)"
  exit 1
fi

spec_tests_dir="$1"
spec_version="${2:-nightly}"
presets="general minimal mainnet"
version_file="${spec_tests_dir}/version.txt"

confirm_replace() {
  local current="$1"
  local expected="$2"
  echo "Reference tests version mismatch"
  echo "  Have: ${current}"
  echo "  Need: ${expected}"
  printf "Delete %s and download %s? [Y/n] " "${spec_tests_dir}" "${expected}"
  read -r reply || true
  case "${reply:-Y}" in
    [yY]|[yY][eE][sS]) rm -rf "${spec_tests_dir}" ;;
    [nN]|[nN][oO]) echo "Aborting."; exit 1 ;;
    *) rm -rf "${spec_tests_dir}" ;;
  esac
  mkdir -p "${spec_tests_dir}"
}

mkdir -p "${spec_tests_dir}"

pinned_run_id=""
nightly_date=""
if [[ "${spec_version}" == nightly* && "${spec_version}" != "nightly" ]]; then
  suffix="${spec_version#nightly-}"
  if [[ "${suffix}" =~ ^[0-9]+$ ]]; then
    pinned_run_id="${suffix}"
  elif [[ "${suffix}" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
    nightly_date="${suffix}"
  else
    echo "Invalid nightly version: ${spec_version}"
    echo "Expected 'nightly', 'nightly-YYYY-MM-DD', or 'nightly-<run_id>'"
    exit 1
  fi
fi

if [[ "${spec_version}" == nightly* ]]; then
  if [[ -z "${GITHUB_TOKEN:-}" ]]; then
    if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
      GITHUB_TOKEN="$(gh auth token)"
      export GITHUB_TOKEN
    else
      echo "Error: GITHUB_TOKEN is not set and gh CLI is not authenticated"
      echo "Either set GITHUB_TOKEN or run: gh auth login"
      exit 1
    fi
  fi

  for cmd in curl jq; do
    if ! command -v "${cmd}" >/dev/null 2>&1; then
      echo "Error: ${cmd} is not installed"
      exit 1
    fi
  done

  repo="${SPEC_REPO:-ethereum/consensus-specs}"
  workflow="tests.yml"
  branch="${SPEC_BRANCH:-master}"
  api="https://api.github.com"
  auth_header="Authorization: token ${GITHUB_TOKEN}"

  if [[ -n "${nightly_date}" ]]; then
    run_json="$(curl -s -H "${auth_header}" \
      "${api}/repos/${repo}/actions/workflows/${workflow}/runs?branch=${branch}&status=success&created=${nightly_date}&per_page=1")"
    run_id="$(echo "${run_json}" | jq -r '.workflow_runs[0].id')"
    if [[ -z "${run_id}" || "${run_id}" == "null" ]]; then
      echo "No successful nightly run found for ${repo} on ${nightly_date} (${workflow} on ${branch})"
      exit 1
    fi
    echo "Resolved ${spec_version} to run ${run_id}"
  elif [[ "${spec_version}" == "nightly" ]]; then
    run_json="$(curl -s -H "${auth_header}" \
      "${api}/repos/${repo}/actions/workflows/${workflow}/runs?branch=${branch}&status=success&per_page=1")"
    run_id="$(echo "${run_json}" | jq -r '.workflow_runs[0].id')"
    if [[ -z "${run_id}" || "${run_id}" == "null" ]]; then
      echo "No successful nightly run found for ${repo} (${workflow} on ${branch})"
      exit 1
    fi
  else
    run_id="${pinned_run_id}"
    run_json="$(curl -sf -H "${auth_header}" \
      "${api}/repos/${repo}/actions/runs/${run_id}")" || {
        echo "Error: could not fetch run ${run_id}"
        exit 1
      }
  fi
  expected_version="nightly-${run_id}"
else
  expected_version="${spec_version}"
fi

if [[ -f "${version_file}" ]]; then
  current_version="$(cat "${version_file}")"
  if [[ "${current_version}" == "${expected_version}" ]]; then
    exit 0
  fi
  confirm_replace "${current_version}" "${expected_version}"
else
  if [[ -d "${spec_tests_dir}/tests" ]] || compgen -G "${spec_tests_dir}/*" >/dev/null; then
    confirm_replace "unknown (missing version.txt)" "${expected_version}"
  fi
fi

if [[ "${spec_version}" != nightly* ]]; then
  for preset in ${presets}; do
    tar_path="${spec_tests_dir}/${preset}.tar.gz"
    if [[ ! -f "${tar_path}" ]]; then
      echo "Downloading: ${spec_version}/${preset}.tar.gz"
      curl --progress-bar --location --show-error --retry 3 --retry-all-errors --fail \
        -o "${tar_path}" \
        "https://github.com/ethereum/consensus-specs/releases/download/${spec_version}/${preset}.tar.gz" || {
          echo "Curl failed. Aborting"
          rm -f "${tar_path}"
          exit 1
        }
    fi
    if [[ ! -d "${spec_tests_dir}/tests/${preset}" ]]; then
      echo "Extracting ${preset}..."
      tar -xzf "${tar_path}" -C "${spec_tests_dir}"
    fi
    rm -f "${tar_path}"
  done
  echo "${expected_version}" > "${version_file}"
  exit 0
fi

for preset in ${presets}; do
  artifact_name="${preset}.tar.gz"

  artifact_url="$(curl -s -H "${auth_header}" \
    "${api}/repos/${repo}/actions/runs/${run_id}/artifacts?name=${artifact_name}&per_page=1" | \
    jq -r '.artifacts[0].archive_download_url')"

  if [[ -z "${artifact_url}" || "${artifact_url}" == "null" ]]; then
    echo "Skipping ${artifact_name} (not found in run ${run_id})"
    continue
  fi

  tar_path="${spec_tests_dir}/${preset}.tar.gz"

  echo "Downloading artifact: ${artifact_name} (run ${run_id})"
  curl --progress-bar --location --show-error --retry 3 --retry-all-errors --fail \
    -H "${auth_header}" -H "Accept: application/vnd.github+json" \
    --output "${tar_path}" "${artifact_url}"

  if [[ ! -d "${spec_tests_dir}/tests/${preset}" ]]; then
    echo "Extracting ${preset}..."
    tar -xzf "${tar_path}" -C "${spec_tests_dir}"
  fi
  rm -f "${tar_path}"
done

echo "${expected_version}" > "${version_file}"
