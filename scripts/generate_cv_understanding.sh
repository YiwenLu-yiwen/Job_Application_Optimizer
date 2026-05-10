#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH="${PYTHONPATH:-src}" python -m job_application_optimizer.resume.understanding \
  "${RESUME_FILE:-data/resume.pdf}" \
  -o "${CV_UNDERSTANDING_FILE:-data/cv_deep_understanding.md}"
