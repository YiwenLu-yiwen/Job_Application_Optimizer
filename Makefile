.PHONY: install install-browser run cv-understanding interview interview-all run-interview run-interview-all

RESUME ?= data/resume.pdf
URLS ?= data/urls.txt
CV_UNDERSTANDING ?= data/cv_deep_understanding.md
OUT ?= outputs

install:
	python -m pip install -r requirements.txt
	python -m pip install -e .

install-browser:
	python -m playwright install chromium

run:
	PYTHONPATH=src python -m job_application_optimizer.cli \
		--urls $(URLS) \
		--resume $(RESUME) \
		--cv-understanding $(CV_UNDERSTANDING) \
		--out $(OUT)

cv-understanding:
	PYTHONPATH=src python -m job_application_optimizer.resume.understanding \
		$(RESUME) \
		-o $(CV_UNDERSTANDING)

interview:
	@if [ -z "$(JOB_FOLDER)" ]; then echo "Usage: make interview JOB_FOLDER=outputs/YYYY-MM-DD/Company_Role"; exit 1; fi
	PYTHONPATH=src python -m job_application_optimizer.generation.interview_prep \
		--job-folder "$(JOB_FOLDER)" \
		--cv-understanding $(CV_UNDERSTANDING) $(if $(OVERWRITE),--overwrite,)

interview-all:
	@if [ -z "$(RUN_DIR)" ]; then echo "Usage: make interview-all RUN_DIR=outputs/YYYY-MM-DD"; exit 1; fi
	PYTHONPATH=src python -m job_application_optimizer.generation.interview_prep \
		--run-dir "$(RUN_DIR)" \
		--cv-understanding $(CV_UNDERSTANDING) $(if $(OVERWRITE),--overwrite,)

run-interview: interview

run-interview-all: interview-all
