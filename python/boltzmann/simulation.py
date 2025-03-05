"""
This module provides some "glue" for easily making scripts which run simulations.
It provides automatic progress logging, checkpointing, and simulation resumption via the command line.
"""

import argparse as ap
import logging
import os
import sys
import datetime

from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Protocol

from boltzmann.core import Device, Simulation
from boltzmann.units import Array1dT
from boltzmann.utils.logger import dotted, tick
from boltzmann.utils.option import Some, to_opt


logger = logging.getLogger(__name__)


@dataclass
class TimeMeta:
    """
    Contains parameters for output interval and simulation duration.
    """

    dt_step: float
    dt_output: float
    t_max: float
    output_count: int

    @property
    def batch_steps(self) -> int:
        return int(self.dt_output / self.dt_step + 1e-8)

    @staticmethod
    def make(
        *,
        dt_step: float,
        dt_output: float | None = None,
        t_max: float | None = None,
        output_count: int | None = None,
    ) -> "TimeMeta":
        d_ = to_opt(dt_output)
        t_ = to_opt(t_max)
        i_ = to_opt(output_count)

        match (d_, t_, i_):
            case (None, Some(t), Some(i)):
                d = t / i
            case (Some(d), None, Some(i)):
                t = i * d
            case (Some(d), Some(t), None):
                i = int(t / d + 1e-8)
            case _:
                raise ValueError(
                    f"Must specify dt_step and exactly two other arguments! [{dt_output=}, {t_max=}, {output_count=}]"
                )

        return TimeMeta(dt_step, d, t, i)


def run_sim(
    sim: Simulation,
    time: TimeMeta,
    output_dir: Path | str,
    checkpoint_interval: int = 1,
) -> Generator[int, None, None]:
    """
    Run the simulation with the given batch configuration, write checkpoints, and provide logging.

    This returns a generator, which for each iteration runs the required number of LBM steps.
    It yields an integer which is the current iteration.

    .. code-block:: python

        for i in run_sim(sim, time_meta, output_dir):
            plt.imshow(sim.fluid.T)
            plt.savefig(output_dir / f"output_{i:04d}.png")

    """

    # Check or create the output directory.
    output_dir = Path(output_dir)
    if output_dir.exists():
        assert output_dir.is_dir(), f"Base path must be a directory! [{output_dir=}]"
    else:
        output_dir.mkdir()

    # Iteration count must be even due to AA pattern.
    batch_iters = 2 * int(time.batch_steps / 2 + 0.5)

    # Log simulatin info.
    dotted(logger, "Output directory", str(output_dir))
    dotted(logger, "Memory usage", f"{sim.size_bytes // 1_000_000:,d}", "MB")
    dotted(logger, "Iters / output", batch_iters)
    dotted(logger, "Cells", f"{sim.cells.count / 1e6:,.1f}M")

    max_i = time.output_count
    sim_i = sim.iteration // batch_iters

    # Immediately yield so we write the first output before anything.
    # We also do this on restart incase of partial output.
    if sim.iteration > 0:
        logger.info(f"Resuming from step {sim_i}")
        yield sim_i
    else:
        yield 0

    timer_tot = tick()
    for i in range(sim_i + 1, max_i):
        timer_sim = tick()
        sim.iterate(batch_iters)

        # Call tock() before checkpointing so it's not included in the mlups calculation.
        perf_sim = timer_sim.tock(events=sim.cells.count * batch_iters)
        mlups = perf_sim.events / (1e6 * perf_sim.seconds)

        # Caller writes their output(s) here.
        timer_out = tick()
        yield i

        # Write checkpoint if requested. (It probably is!)
        if (i % checkpoint_interval) == 0:
            # First write to a temp file, then move into place atomically to avoid corruption.
            sim.write_checkpoint(str(output_dir / "checkpoint.mpk.tmp"))
            os.replace(output_dir / "checkpoint.mpk.tmp", output_dir / "checkpoint.mpk")

        perf_out = timer_out.tock()

        # Get total elapsed time and ETA.
        t = timer_tot.elapsed
        t = str(t).split(".")[0]
        per = timer_tot.elapsed / (i - sim_i)
        rem = max_i - i
        eta = datetime.datetime.now() + rem * per

        logger.info(
            f"Batch {i}: {t} {mlups:5,.0f} MLUP/s, sim {perf_sim.seconds:.1f}s, out {perf_out.seconds:.1f}s, eta {eta:%H:%M:%S}"
        )


def make_sim(
    cnts: Array1dT,
    omega_ns: float,
    dir: Path | str = "out",
    device: Device = "gpu",
    q: int | None = None,
) -> Simulation:
    """
    Convenience function to get a new :py:class:`Simulation` object or a previously checkpointed one.

    * Allows for resuming a simulation simply by calling the script with the `--resume` flag.
    * Allows setting the OpenCL device to run on with the `--device` option.

    Apart from for very simple scripts or examples this is probably the "constructor" you want.

    This either constructs a `Simulation` with the given arguments or it loads one from the
    previously saved checkpoint, depending on the command line arguments.
    In the case of loading, the passed arguments are ignored and the serialized values are used.
    In the case of constructing, you will want to set the initial values.
    The two cases can be distinguished by checking if `sim.iterations == 0` or not.

    .. code-block:: bash

        # An example script:
        $ cat my_sim.py
        out = "out"
        sim = make_runner([100, 100], 1 / 0.51, out)

        if sim.iterations == 0:
            # set initial conditions
            ...

        time = TimeMeta.make(dt_step=0.01, dt_output=0.1, output_count=100)
        for iter in run_sim(sim, time, out):
            write_output(iter)  # for example

        # Use the CLI to resume a previously started simulation:
        $ python my_sim.py --device cpu --resume
    """

    # If it looks like we're in an ipython session, ignore command line arguments.
    try:
        get_ipython()  # noqa: F821
        args = []
    except NameError:
        args = sys.argv[1:]

    parser = ap.ArgumentParser()
    # fmt: off
    parser.add_argument("--device", choices={"gpu", "cpu"}, default=device, help="The OpenCL device to run on")
    parser.add_argument("--resume", action="store_true", help="Resume from the checkpoint.")
    parser.add_argument("--out-dir", type=Path, default=dir, help="Path to the directory where the checkpoint is stored.")
    # fmt: on
    args = parser.parse_args(args)

    # Either create the sim or load it.
    if args.resume:
        sim = Simulation.load_checkpoint(args.device, str(args.out_dir / "checkpoint.mpk"))
    else:
        sim = Simulation(args.device, cnts, omega_ns, q=q)

    return sim


# PONDER: I can see some utility in having versions of the 'make_sim' and 'run_sim' functions,
#         which accept `Initializer` and `Outputter`, respectively. Namely it would allow for
#         common output code to be run for various simulations. However, I'm not sure if this is
#         ever something we will want in practice.


class Initializer(Protocol):
    """
    Implementations of this are responsible for setting the initial conditions of the simulation.
    """

    def __call__(self, sim: Simulation): ...


class Outputter(Protocol):
    """
    Implementations of this are responsible for generating & saving the simuation output,
    i.e. any graphs or visualizations.

    Note, this is distinct from checkpointing which is the responsiblity of the `SimulationRunner`.
    """

    def __call__(self, sim: Simulation, output_dir: Path, iter: int): ...


class SimulationScript:
    """
    Simple class which lets us build simulation scripts via decorators on functions.

    Under the hood it calls :py:meth:`make_sim` and :py:meth:`run_sim` so the behaviour is the same.
    But this saves us some wiring of command line arguments into the function calls.
    The difference is purely in the syntax of how the simulation is configured & set up.

    .. code-block:: python

        # The simulation will run automatically when the with-block is exited.
        with SimulationScript(...) as sim:

            @sim.init
            def init(sim: Simulation):
                # Set some initial conditions.
                pass

            @sim.out
            def out(output_dir: Path, iter: int):
                # Save some output.
                pass
    """

    def __init__(
        self, omega_ns: float, time_meta: TimeMeta, output_dir: str | Path, q: int | None = None
    ) -> None:
        self.sim = make_sim([100, 100], dir=output_dir, omega_ns=omega_ns, q=q)
        self.time_meta = time_meta
        self.output_dir = Path(output_dir)
        self.initf = None
        self.outf = None

    def init(self, init: Initializer) -> "SimulationScript":
        """
        Set the :py:meth:`Initializer` for the simulation.

        Intended to be used as a decorator (see :py:class:`SimulationScript`.)

        :param init: The :py:class:`Initializer` function which will set the simulation inital conditions.
        """
        self.initf = init
        return self

    def out(self, out: Outputter) -> "SimulationScript":
        """
        Set the :py:meth:`Outputter` for the simulation.

        Intended to be used as a decorator (see :py:class:`SimulationScript`.)

        :param out: The :py:class:`Outputter` function which will generate the simulation output.
        """
        self.outf = out
        return self

    def run(self):
        """
        Run the pre-configured simulation.
        See :py:meth:`run_sim` for more details.
        """
        if self.initf is None:
            raise ValueError("No initialization function is set.")

        if self.outf is None:
            raise ValueError("No output function is set.")

        # Initialize the simulation if needed.
        if self.sim.iteration == 0:
            self.initf(self.sim)

        # Run it.
        for i in run_sim(self.sim, self.time_meta, self.output_dir):
            self.outf(self.sim, self.output_dir, i)

    def __enter__(self) -> "SimulationScript":
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.run()
