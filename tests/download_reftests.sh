#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <spec_tests_dir> [spec_version]"
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

if [[ "${spec_version}" == nightly* ]]; then
  if [[ -z "${GITHUB_TOKEN:-}" ]]; then
    echo "Error: GITHUB_TOKEN is not set"
    exit 1
  fi

  for cmd in curl jq unzip; do
    if ! command -v "${cmd}" >/dev/null 2>&1; then
      echo "Error: ${cmd} is not installed"
      exit 1
    fi
  done

  repo="${CONSENSUS_SPECS_REPO:-ethereum/consensus-specs}"
  workflow="${CONSENSUS_SPECS_WORKFLOW:-nightly-reftests.yml}"
  branch="${CONSENSUS_SPECS_BRANCH:-master}"
  api="https://api.github.com"
  auth_header="Authorization: token ${GITHUB_TOKEN}"

  if [[ "${spec_version}" == "nightly" ]]; then
    run_id="$(curl -s -H "${auth_header}" \
      "${api}/repos/${repo}/actions/workflows/${workflow}/runs?branch=${branch}&status=success&per_page=1" | \
      jq -r '.workflow_runs[0].id')"

    if [[ -z "${run_id}" || "${run_id}" == "null" ]]; then
      echo "No successful nightly workflow run found for ${repo} (${workflow} on ${branch})"
      exit 1
    fi
    expected_version="nightly-${run_id}"
  else
    run_id="${spec_version#nightly-}"
    if [[ -z "${run_id}" || "${run_id}" == "${spec_version}" ]]; then
      echo "Invalid nightly version: ${spec_version} (expected nightly-<run_id>)"
      exit 1
    fi
    expected_version="${spec_version}"
  fi
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
  case "${preset}" in
    minimal) artifact_name="Minimal Test Configuration" ;;
    mainnet) artifact_name="Mainnet Test Configuration" ;;
    general) artifact_name="General Test Configuration" ;;
    *)
      echo "Unsupported preset: ${preset} (expected: minimal|mainnet|general)"
      exit 1
      ;;
  esac

  artifact_url="$(curl -s -H "${auth_header}" \
    "${api}/repos/${repo}/actions/runs/${run_id}/artifacts" | \
    jq -r --arg name "${artifact_name}" '.artifacts[] | select(.name == $name) | .archive_download_url' | \
    head -n 1)"

  if [[ -z "${artifact_url}" || "${artifact_url}" == "null" ]]; then
    echo "Artifact not found: ${artifact_name} (run ${run_id})"
    exit 1
  fi

  zip_path="${spec_tests_dir}/${preset}.zip"

  echo "Downloading artifact: ${artifact_name} (run ${run_id})"
  curl --progress-bar --location --show-error --retry 3 --retry-all-errors --fail \
    -H "${auth_header}" -H "Accept: application/vnd.github+json" \
    --output "${zip_path}" "${artifact_url}"

  unzip -qo "${zip_path}" -d "${spec_tests_dir}"
  rm -f "${zip_path}"

  tar_path="${spec_tests_dir}/${preset}.tar.gz"
  if [[ ! -f "${tar_path}" ]]; then
    echo "Expected ${tar_path} after download, but it was not found"
    exit 1
  fi
  if [[ ! -d "${spec_tests_dir}/tests/${preset}" ]]; then
    echo "Extracting ${preset}..."
    tar -xzf "${tar_path}" -C "${spec_tests_dir}"
  fi
  rm -f "${tar_path}"
done

echo "${expected_version}" > "${version_file}"
