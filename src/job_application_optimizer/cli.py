"""Command-line entrypoint for the batch optimizer."""

from job_application_optimizer.pipeline.batch_runner import main


__all__ = ["main"]


if __name__ == "__main__":
    main()
