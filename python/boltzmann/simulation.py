import argparse as ap
import logging
import os
import sys
import datetime

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from boltzmann.core import Simulation, TimeMeta
from boltzmann.utils.logger import PerfInfo, dotted, tick, time


logger = logging.getLogger(__name__)


class OutputCallback(Protocol):
    """
    Implementations of this are responsible for generating & saving the simuation output,
    i.e. any graphs or visualizations.

    Note, this is distinct from checkpointing which is the responsiblity of the `SimulationRunner`.
    """

    def __call__(self, base: Path, iter: int): ...


def run_sim(
    base: Path,
    meta: TimeMeta,
    sim: Simulation,
    outf: OutputCallback,
    *,
    write_checkpoints: bool = True,
):
    if base.exists():
        assert base.is_dir(), f"Base path must be a directory! [{base=}]"
    else:
        base.mkdir()

    cell_count = np.prod(sim.fluid.rho.shape)
    batch_iters = meta.batch_steps
    batch_iters = 2 * int(batch_iters / 2 + 0.5)  # iters must be even due to AA pattern

    dotted(logger, "Output directory", str(base))
    dotted(logger, "Memory usage", f"{sim.size_bytes // 1_000_000:,d}MB")
    dotted(logger, "Iters / output", batch_iters)
    dotted(logger, "Cells", f"{cell_count / 1e6:,.1f}M")

    max_i = meta.output_count
    sim_i = sim.iteration // batch_iters

    if sim.iteration > 0:
        logger.info(f"Resuming from step {sim_i}")

    with time(logger, f"Wrote step {sim_i} outputs."):
        outf(base, sim_i)

    timer_total = tick()

    perf_total = PerfInfo()
    for i in range(sim_i + 1, max_i):
        timer = tick()

        with time(logger, "iterating"):
            sim.iterate(batch_iters)

        # Logging & checkpointing follow.

        # Call tock() before checkpointing so it's not included in the mlups calculation.
        perf_batch = timer.tock(events=cell_count * batch_iters)
        perf_total = perf_total + perf_batch

        mlups_batch = perf_batch.events / (1e6 * perf_batch.seconds)
        mlups_total = perf_total.events / (1e6 * perf_total.seconds)

        outf(base, i)

        if write_checkpoints:
            with time(logger, "checkpointing"):
                # First write to a temp file, then move into place atomically to avoid corruption.
                sim.write_checkpoint(str(base / "checkpoint.mpk.tmp"))
                os.replace(base / "checkpoint.mpk.tmp", base / "checkpoint.mpk")

        perf_out = timer.tock() - perf_batch

        t = timer_total.elapsed
        t = str(t).split(".")[0]

        per = timer_total.elapsed / (i - sim_i)
        rem = max_i - i
        eta = datetime.datetime.now() + rem * per

        logger.info(
            f"Batch {i}:  {t}  {mlups_batch:5,.0f} MLUP/s, sim {perf_batch.seconds:.1f}s, out {perf_out.seconds:.1f}s, eta {eta:%H:%M:%S}"
        )


@dataclass
class SimCLI:
    """
    Command line arguments to confuring running of the simulation.
    """

    dev: str
    base: Path
    resume: bool
    no_checkpoint: bool


def parse_cli(
    base: str | Path = "out",
) -> SimCLI:
    """
    The 'main' method for running sims and handling cmd line args.
    """

    # If it looks like we're in an ipython session, ignore command line arguments.
    try:
        get_ipython()  # noqa: F821
        args = []
    except NameError:
        args = sys.argv[1:]

    parser = ap.ArgumentParser()
    parser.add_argument("--dev", choices={"gpu", "cpu"}, default="gpu")
    parser.add_argument("--base", type=str, default=base)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-checkpoint", action="store_true")
    args = parser.parse_args(args)

    base = Path(args.base)

    return SimCLI(
        args.dev,
        base,
        args.resume,
        args.no_checkpoint,
    )
