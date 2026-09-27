# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import multiprocessing as mp
import multiprocessing.connection
import pprint
import sys
from pathlib import Path

import yaml

from app.scaffold import main as app_main
from src.utils.distributed import init_distributed

parser = argparse.ArgumentParser()
parser.add_argument("--fname", type=str, help="name of config file to load", default="configs.yaml")
parser.add_argument(
    "--devices",
    type=str,
    nargs="+",
    default=["cuda:0", "cuda:1", "cuda:2", "cuda:3", "cuda:4", "cuda:5", "cuda:6", "cuda:7"],
    help="which devices to use on local machine",
)
parser.add_argument(
    "--debugmode",
    type=bool,
    default=False,
    help="Setting this to true will not spin up new processes. "
    "The main code runs the main process, which makes it easier to \
    debug with checkpointing.",
)


def process_main(rank, fname, world_size, devices):
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = str(devices[rank].split(":")[-1])

    import logging

    from src.utils.logging import get_logger

    logger = get_logger(force=True)
    if rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    logger.info(f"called-params {fname}")

    # Load config
    params = None
    with open(fname, "r") as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        logger.info("loaded params...")

    # Log config
    if rank == 0:
        pprint.PrettyPrinter(indent=4).pprint(params)
        folder = params["folder"]
        params_path = os.path.join(folder, "params-pretrain.yaml")
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        with open(params_path, "w") as f:
            yaml.dump(params, f)

    # Init distributed (access to comm between GPUS on same machine)
    world_size, rank = init_distributed(rank_and_world_size=(rank, world_size))
    logger.info(f"Running... (rank: {rank}/{world_size})")

    # Launch the app with loaded config
    app_main(params["app"], args=params)


def launch(fname, devices, target=process_main):
    """Run one process per device and wait for them. Returns the ranks that failed, so the
    launcher can fail when any rank does. Once a rank fails, the others are stopped instead
    of waiting on collectives that rank will never join; ranks stopped this way are not
    reported as failures."""
    processes = [mp.Process(target=target, args=(rank, fname, len(devices), devices)) for rank in range(len(devices))]
    for p in processes:
        p.start()
    running, stopped = set(range(len(processes))), set()
    while running:
        mp.connection.wait([processes[rank].sentinel for rank in running])
        finished = {rank for rank in running if not processes[rank].is_alive()}
        running -= finished
        if any(processes[rank].exitcode != 0 for rank in finished - stopped):
            for rank in running - stopped:
                processes[rank].terminate()
                stopped.add(rank)
    for p in processes:
        p.join()
    return [rank for rank, p in enumerate(processes) if p.exitcode != 0 and rank not in stopped]


if __name__ == "__main__":
    args = parser.parse_args()
    if args.debugmode:
        process_main(rank=0, fname=args.fname, world_size=1, devices=["cuda:0"])
    else:
        mp.set_start_method("spawn")
        failed = launch(args.fname, args.devices)
        if failed:
            sys.exit(f"Training failed on rank(s) {failed}; the other ranks were stopped.")
