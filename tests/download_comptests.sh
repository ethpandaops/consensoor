#!/usr/bin/env bash
# Fetch fork-choice compliance test fixtures from consensus-specs `comptests.yml`.
# Artifact layout extracts to: <dir>/tests/<preset>/<fork>/fork_choice_compliance/...
set -Eeuo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <spec_tests_dir> [spec_version]"
  echo "  spec_version: 'nightly' or 'nightly-<run_id>'"
  exit 1
fi

spec_tests_dir="$1"
spec_version="${2:-nightly}"
version_file="${spec_tests_dir}/comptests-version.txt"

mkdir -p "${spec_tests_dir}"

if [[ -z "${GITHUB_TOKEN:-}" ]]; then
  if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
    GITHUB_TOKEN="$(gh auth token)"
    export GITHUB_TOKEN
  else
    echo "Error: GITHUB_TOKEN is not set and gh CLI is not authenticated"
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
workflow="comptests.yml"
branch="${SPEC_BRANCH:-master}"
api="https://api.github.com"
auth_header="Authorization: token ${GITHUB_TOKEN}"

if [[ "${spec_version}" == "nightly" ]]; then
  run_id="$(curl -s -H "${auth_header}" \
    "${api}/repos/${repo}/actions/workflows/${workflow}/runs?branch=${branch}&status=success&per_page=1" | \
    jq -r '.workflow_runs[0].id')"
  if [[ -z "${run_id}" || "${run_id}" == "null" ]]; then
    echo "No successful ${workflow} run found for ${repo} on ${branch}"
    exit 1
  fi
elif [[ "${spec_version}" == nightly-* ]]; then
  run_id="${spec_version#nightly-}"
  if [[ ! "${run_id}" =~ ^[0-9]+$ ]]; then
    echo "Invalid version: ${spec_version} (expected nightly or nightly-<run_id>)"
    exit 1
  fi
else
  echo "Invalid version: ${spec_version}"
  exit 1
fi

expected_version="nightly-${run_id}"

if [[ -f "${version_file}" ]] && [[ "$(cat "${version_file}")" == "${expected_version}" ]]; then
  exit 0
fi

artifact_name="small.tar.gz"
artifact_url="$(curl -s -H "${auth_header}" \
  "${api}/repos/${repo}/actions/runs/${run_id}/artifacts?name=${artifact_name}&per_page=1" | \
  jq -r '.artifacts[0].archive_download_url')"

if [[ -z "${artifact_url}" || "${artifact_url}" == "null" ]]; then
  echo "Artifact ${artifact_name} not found in run ${run_id}"
  exit 1
fi

tar_path="${spec_tests_dir}/comptests-small.tar.gz"
echo "Downloading compliance artifact: ${artifact_name} (run ${run_id})"
curl --progress-bar --location --show-error --retry 3 --retry-all-errors --fail \
  -H "${auth_header}" -H "Accept: application/vnd.github+json" \
  --output "${tar_path}" "${artifact_url}"

echo "Extracting compliance tests..."
# The tarball lays out `tests/<preset>/<fork>/fork_choice_compliance/...` — merges cleanly
# alongside reftests already in the same dir.
tar -xzf "${tar_path}" -C "${spec_tests_dir}"
rm -f "${tar_path}"
echo "${expected_version}" > "${version_file}"
