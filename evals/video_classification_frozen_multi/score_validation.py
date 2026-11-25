# evals/video_classification_frozen_multi/score_validation.py

"""Score the validation split with a finished multi-view RVSP probe's selected epoch and head, and nothing else.

The probe's test step (`--test_only`) writes per-study validation predictions only while it scores the test
split. A comparison that must stay on validation (two encoders compared by per-study validation MAE with a
paired bootstrap) needs those predictions without opening the test split. This entry point runs the probe's own
`main()` in its validation-scoring mode: the probe is rebuilt from its config, checked against the fingerprint
in `best.pt` (the number of GPUs aside), its head selected and every validation study scored once, exactly as
`run_test` does. The test manifest is never opened, and nothing is sent to wandb.

    python -m evals.video_classification_frozen_multi.score_validation --fname <probe config> --devices cuda:0

Writes `val_selected_predictions.csv` and `val_selected_metrics.json` to the probe's folder.
"""

import argparse
import functools
import logging
import multiprocessing as mp
import os
import sys

import yaml

from evals.video_classification_frozen_multi import eval as multi
from src.utils.distributed import close_distributed, init_distributed

logger = logging.getLogger()

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("--fname", type=str, required=True, help="the probe's config file")
parser.add_argument("--devices", type=str, nargs="+", default=["cuda:0"], help="one process per device")
parser.add_argument("--overwrite", action="store_true", help="replace outputs of an earlier scoring")
parser.add_argument("--tolerance", type=float, default=multi.DEFAULT_SCORE_TOLERANCE,
                    help="largest difference, in mmHg, between the recomputed and the stored validation MAE")


def score_validation(params, overwrite=False, tolerance=multi.DEFAULT_SCORE_TOLERANCE):
    """`params`: the probe's config as written. Returns the metrics on rank 0, `None` on the others."""
    if params.get("eval_name", multi.EVAL_NAME) != multi.EVAL_NAME:
        raise ValueError(f"This scores `{multi.EVAL_NAME}` probes, not `{params.get('eval_name')}`.")
    return multi.main(dict(params, score_validation={"overwrite": overwrite, "tolerance": tolerance}))


def process_main(args, rank, fname, world_size, devices):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(devices[rank].split(":")[-1])
    logging.basicConfig()
    logger.setLevel(logging.INFO if rank == 0 else logging.ERROR)
    with open(fname) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    init_distributed(rank_and_world_size=(rank, world_size))
    score_validation(params, overwrite=args.overwrite, tolerance=args.tolerance)
    close_distributed()


def launch_ranks(args):
    """One process per device, as `evals.main`: once a rank fails the others are stopped, and the command
    exits non-zero."""
    from app.main import launch

    failed = launch(args.fname, args.devices, target=functools.partial(process_main, args))
    if failed:
        sys.exit(f"Validation scoring failed on rank(s) {failed}; the other ranks were stopped.")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    launch_ranks(parser.parse_args())
