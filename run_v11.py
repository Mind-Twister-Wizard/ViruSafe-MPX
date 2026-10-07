from __future__ import annotations

import sys
import traceback
from pathlib import Path

from src_v11.core import V11State, load_yaml, setup_logger
from src_v11.pipeline import run_all


def main():
    root=Path(__file__).resolve().parent
    cfg=load_yaml(root/"config_v1.1.yaml")
    logger=setup_logger(root)
    state=V11State(root/cfg["paths"]["state_file"])
    try:
        run_all(root,cfg,state,logger)
        return 0
    except KeyboardInterrupt:
        logger.warning("Interrupted by user. Completed v1.1 caches/stages are preserved; rerun the same BAT to resume.")
        return 130
    except Exception as e:
        logger.error("ViruSafe-MPX v1.1 failed: %s",e)
        logger.error(traceback.format_exc())
        logger.error("Completed v1.0.4 results and v1.1 caches are preserved. Correct the issue and rerun the same BAT.")
        return 1

if __name__=="__main__":
    raise SystemExit(main())
