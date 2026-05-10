#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${JOB_FOLDER:-}" ]]; then
  PYTHONPATH="${PYTHONPATH:-src}" python -m job_application_optimizer.generation.interview_prep \
    --job-folder "${JOB_FOLDER}" \
    --cv-understanding "${CV_UNDERSTANDING_FILE:-data/cv_deep_understanding.md}" \
    ${OVERWRITE:+--overwrite}
elif [[ -n "${RUN_DIR:-}" ]]; then
  PYTHONPATH="${PYTHONPATH:-src}" python -m job_application_optimizer.generation.interview_prep \
    --run-dir "${RUN_DIR}" \
    --cv-understanding "${CV_UNDERSTANDING_FILE:-data/cv_deep_understanding.md}" \
    ${OVERWRITE:+--overwrite}
else
  echo "Usage: JOB_FOLDER=outputs/YYYY-MM-DD/Company_Role scripts/generate_interview_prep.sh"
  echo "   or: RUN_DIR=outputs/YYYY-MM-DD scripts/generate_interview_prep.sh"
  exit 1
fi
