from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from src.pipeline import run_pipeline
from src.utils import PipelineState, ensure_dirs, get_device, load_config, seed_everything, setup_logging, system_info


def main():
    parser = argparse.ArgumentParser(description="ViruSafe-MPX resumable research pipeline")
    parser.add_argument("--resume", action="store_true", help="Resume from existing state/checkpoints (default behavior).")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    cfg = load_config(root / "config.yaml")
    paths = ensure_dirs(root, cfg)
    logger = setup_logging(paths["logs"] / "pipeline_python.log")
    seed_everything(int(cfg["project"]["seed"]))
    device = get_device(str(cfg["project"].get("device", "auto")))
    state = PipelineState(paths["state"] / "pipeline_state.json")

    logger.info("ViruSafe-MPX started")
    logger.info("Device: %s", device)
    logger.info("System: %s", system_info(device))
    if device.type == "cpu":
        logger.warning("CUDA GPU was not detected. The pipeline will still run, but training will be much slower.")

    try:
        result = run_pipeline(cfg, paths, state, device, logger)
        logger.info("ViruSafe-MPX completed successfully.")
        logger.info("Report: %s", paths["outputs"] / "report" / "index.html")
        return 0
    except KeyboardInterrupt:
        logger.warning("Interrupted by user. Checkpoints/state are preserved; rerun the BAT to resume.")
        return 130
    except Exception as e:
        logger.error("Pipeline failed: %s", e)
        logger.error(traceback.format_exc())
        logger.error("Completed downloads/checkpoints are preserved. Rerun RUN_VIRUSAFE_MPX.bat to resume after correcting the issue.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
