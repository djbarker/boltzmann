"""
This module provides some "glue" for easily making scripts which run simulations.
It provides automatic progress logging, checkpointing, and simulation resumption via the command line.

See :doc:`guides/script <guides/script>` for more information on how to use this module.
"""

import argparse as ap
import logging
import numpy as np
import os
import re
import sys
import datetime

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Protocol

from boltzmann.core import Device, Simulation
from boltzmann.units import Array1dT
from boltzmann.utils.logger import dotted, tick
from boltzmann.utils.option import Some, to_opt


logger = logging.getLogger(__name__)


@dataclass
class IterInfo:
    """
    Contains the simulation output interval and count.

    The output interval is the number of LBM iterations needed per output timestep.
    If one timestep of the LBM corresponds to ``dt`` and we wish to output every ``dt_output``,
    then the number of LBM iterations per output is ``dt_output / dt``.
    """

    interval: int  #: LBM iterations per output.
    count: int  #: Total number of outputs.

    @staticmethod
    def make(
        *,
        dt: float,
        dt_output: float | None = None,
        t_max: float | None = None,
        count: int | None = None,
    ) -> "IterInfo":
        """
        Construct a :py:class:`IterInfo` object from a valid combination of the parameters.
        Valid combinations are those that are sufficient to infer the others.

        Specific valid combinations are

        * ``dt``, ``dt_output``, ``count`` - ``interval`` is inferred as ``dt_output / dt``.
        * ``dt``, ``dt_output``, ``t_max`` - ``count`` is inferred as ``t_max / dt_output``, ``interval`` is inferred as above.
        * ``dt``, ``t_max``, ``count`` - ``interval`` is inferred as ``(t_max / count) / dt``.

        :param dt: The physical time corresponding to each iteration of the LBM simulation.
        :param dt_output: The elapsed physical time after which we wish to produce each output.
        :param t_max: The total physical time to run the simulation for.
        :param count: The total number of outputs to produce.
        """
        d_ = to_opt(dt_output)
        t_ = to_opt(t_max)
        i_ = to_opt(count)

        match (d_, t_, i_):
            case (None, Some(t), Some(i)):
                d = t / i
            case (Some(d), None, Some(i)):
                t = i * d
            case (Some(d), Some(t), None):
                i = int(t / d + 1e-8)
            case _:
                raise ValueError(
                    f"Must specify dt and exactly two other arguments! [{dt_output=}, {t_max=}, {count=}]"
                )

        return IterInfo(int(d / dt), i)


class CheckpointGater(ABC):
    """
    Tell :py:meth:`run_sim` when it should write a checkpoint.
    """

    @abstractmethod
    def allow(self) -> bool:
        """
        Returns true if the simulation should write a checkpoint.
        """
        pass


@dataclass
class EveryN(CheckpointGater):
    """
    Allow a checkpoint to be written every N times the :py:meth:`CheckpointGater.allow` method is called.
    """

    interval: int
    _curr: int = field(default=0)

    def __post_init__(self):
        assert self.interval > 0, "Checkpoint Interval must be positive!"

    def allow(self) -> bool:
        self._curr = (self._curr + 1) % self.interval
        return self._curr == 1


@dataclass
class EveryT(CheckpointGater):
    """
    Allow a checkpoint to be written at regular time intervals.
    """

    interval: datetime.timedelta
    _last: datetime.datetime = field(default=datetime.datetime(1970, 1, 1))

    def allow(self) -> bool:
        now = datetime.datetime.now()
        if now - self._last > self.interval:
            self._last = now
            return True
        return False


def parse_checkpoints(s: str) -> CheckpointGater:
    """
    Parse a string into a :py:class:`CheckpointGater` object.

    There are only two valid formats:

        * ``"<N>"`` - which will checkpoint every ``N`` iterations.
        * ``"<N>m"``- which will checkpoint every ``N`` minutes.

    :param s: The string to parse.
    :returns: A :py:class:`CheckpointGater`.
    """

    if (m := re.match(r"^(\d+)$", s)) is not None:
        return EveryN(int(m.group(1)))
    elif (m := re.match(r"^(\d+)m$", s)) is not None:
        return EveryT(datetime.timedelta(minutes=int(m.group(1))))
    else:
        raise ValueError(f"Unable to parse checkpoint interval! [{s!r}]")


def run_sim(
    sim: Simulation,
    meta: IterInfo,
    output_dir: Path | str,
    checkpoints: CheckpointGater | None = None,
) -> Generator[int, None, None]:
    """
    Run the simulation with the given batch configuration, write checkpoints, and provide logging.

    This returns a generator, which for each iteration runs the required number of LBM steps.
    It yields an integer which is the current iteration.

    .. code-block:: python

        for i in run_sim(sim, meta, output_dir):
            plt.imshow(sim.fluid.rho.T)
            plt.savefig(output_dir / f"output_{i:04d}.png")

    Once the generator resumes it outputs the checkpoint file in the specified directory with the name ``checkpoint.mpk``.
    It is first written to a temporary file then moved into place to avoid possible corruption.
    The frequency of checkpointing is controlled by the ``checkpoints`` argument.

    :param sim: The core Simulation object.
    :param meta: The steps/output and output count info.
    :param output_dir: The directory to write the checkpoint to.
    :param checkpoints: The checkpointing frequency.
    """

    # Check or create the output directory.
    output_dir = Path(output_dir)
    if output_dir.exists():
        assert output_dir.is_dir(), f"Base path must be a directory! [{output_dir=}]"
    else:
        output_dir.mkdir()

    checkpoints = checkpoints or EveryN(5)

    # Iteration count must be even due to AA pattern.
    batch_iters = 2 * int(meta.interval / 2 + 0.5)

    # Log simulatin info.
    dotted(logger, "OpenCL device", sim.device_info.replace("Corporation ", ""))
    dotted(logger, "Output directory", str(output_dir))
    dotted(logger, "Memory usage", f"{sim.size_bytes // 1_000_000:,d}", "MB")
    dotted(logger, "Iters / output", batch_iters)
    dotted(logger, "Cells", f"{sim.cells.count / 1e6:,.1f}M")
    dotted(logger, "     ", " x ".join([str(int(i)) for i in sim.cells.counts]))

    max_i = meta.count
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

        if np.any(~np.isfinite(sim.fluid.vel)):
            raise ValueError("Non-finite value detected!")

        # Call tock() before checkpointing so it's not included in the mlups calculation.
        perf_sim = timer_sim.tock(events=sim.cells.count * batch_iters)
        mlups = perf_sim.events / (1e6 * perf_sim.seconds)

        # Caller writes their output(s) here.
        timer_out = tick()
        yield i

        # Write checkpoint if requested.
        if checkpoints.allow():
            # First write to a temp file, then move into place atomically to avoid corruption.
            sim.write_checkpoint(str(output_dir / "checkpoint.mpk.tmp"))
            os.replace(output_dir / "checkpoint.mpk.tmp", output_dir / "checkpoint.mpk")
            logger.info("Checkpoint written.")

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


@dataclass
class SimulationArgs:
    out_dir: Path
    device: Device
    resume: bool
    checkpoints: CheckpointGater


def parse_args(out_dir: Path | str, device: Device = "gpu") -> SimulationArgs:
    """
    Parse the standard command line arguments for :py:class:`SimulationScript`.

    The options from the command line are always given preference but you can specify the defaults via the function's arguments.
    """
    # If it looks like we're in an ipython session, ignore command line arguments.
    try:
        get_ipython()  # type: ignore
        args = []
    except NameError:
        args = sys.argv[1:]

    parser = ap.ArgumentParser()
    # fmt: off
    parser.add_argument("--device", choices={"gpu", "cpu"}, default=device, help="The OpenCL device to run on")
    parser.add_argument("--resume", action="store_true", help="Resume from the checkpoint.")
    parser.add_argument("--out-dir", type=Path, default=out_dir, help="Path to the directory where the checkpoint is stored.")
    parser.add_argument("--checkpoints", type=str, default="5", help="How often to write checkpoints.")
    # fmt: on
    args = parser.parse_args(args)

    return SimulationArgs(
        Path(args.out_dir),
        args.device,
        args.resume,
        parse_checkpoints(args.checkpoints),
    )


# PONDER: I can see some utility in having versions of the 'make_sim' and 'run_sim' functions,
#         which accept `Initializer` and `Outputter`, respectively. Namely it would allow for
#         common output code to be run for various simulations. However, I'm not sure if this is
#         ever something we will want in practice.


class Initializer(Protocol):
    """
    Implementations of this are responsible for setting the initial conditions of the simulation.
    """

    def __call__(self): ...


class Outputter(Protocol):
    """
    Implementations of this are responsible for generating & saving the simuation output,
    i.e. any graphs or visualizations.

    Note, this is distinct from checkpointing which is the responsiblity of the `SimulationRunner`.
    """

    def __call__(self, output_dir: Path, iter: int): ...


class SimulationScript:
    """
    Allows us to easily build scripts that handle the initialization, output, running & checkpointing of the simulation.

    This either constructs a new :py:class:`Simulation` or loads one from the previously saved checkpoint,
    depending on the command line arguments passed.

    - In the case of constructing, the passed ``count`` and ``omega`` values are used, then the ``init`` function will be called to set the initial conditions.
    - In the case of loading, the passed arguments are ignored and the deserialized values are used. The ``init`` function is not called.

    Under the hood it calls :py:meth:`run_sim` so the behaviour is the same.
    However, this saves us worrying about manually loading checkpoints & wiring the command line arguments into the function calls.

    .. code-block:: python

        from boltzmann.simulation import SimulationScript

        # The simulation will run automatically when the with-block is exited.
        # `sim` is a Simulation object.
        with (script := SimulationScript(...)) as sim:

            @script.init
            def init():
                # Set some initial conditions.
                pass

            @script.out
            def out(output_dir: Path, iter: int):
                # Save some output.
                pass
    """

    def __init__(
        self,
        counts: Array1dT,
        omega_ns: float,
        meta: IterInfo,
        output_dir: str | Path,
        q: int | None = None,
    ) -> None:
        args = parse_args(output_dir)

        # Either create the sim or load it.
        if args.resume:
            sim = Simulation.load_checkpoint(args.device, str(args.out_dir / "checkpoint.mpk"))
        else:
            sim = Simulation(args.device, counts, omega_ns, q=q)

        self.args = args
        self.sim = sim
        self.meta = meta
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
            self.initf()

        # Run it.
        for i in run_sim(self.sim, self.meta, self.args.out_dir, self.args.checkpoints):
            self.outf(self.args.out_dir, i)

    def __enter__(self) -> "Simulation":
        return self.sim

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            return False
        self.run()
