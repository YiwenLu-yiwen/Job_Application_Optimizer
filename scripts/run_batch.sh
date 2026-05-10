#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH="${PYTHONPATH:-src}" python -m job_application_optimizer.cli \
  --urls "${URLS_FILE:-data/urls.txt}" \
  --resume "${RESUME_FILE:-data/resume.pdf}" \
  --cv-understanding "${CV_UNDERSTANDING_FILE:-data/cv_deep_understanding.md}" \
  --out "${OUTPUT_DIR:-outputs}"
