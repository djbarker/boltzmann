Simulation Scripts
==================

The example shown in :doc:`guides/basic <basic>` shows the core of getting a simulation running,
but it is lacking some common niceties that you would want for a real simulation.
Specifically for a longer-running simulation you probably want it to:

- Produce outputs for multiple time-steps.
- Provide some logging to indicate progress.
- Save checkpoints at regular intervals.
- Resume from a checkpoint if it is interrupted.

Rather than re-invent the wheel for every simulation the :py:mod:`boltzmann.simulation` 
module provides us with utilities to achieve all of these quickly.

There are two ways to use this module (and I'm not settled on which I prefer).

1. Imperative
-------------

We can wire up a script quickly using two useful functions :py:meth:`parse_args` and :py:meth:`run_sim`.
The methods are explained just below but first, here's an example

.. code-block:: python

    from boltzmann.core import Simulation, bgk
    from boltzmann.simulation import parse_args, run_sim, IterInfo
    
    # Parse the standard command line arguments.
    args = parse_args()  

    if args.resume:
        # Load previously started simulation.
        sim = Simulation.load_checkpoint(args.device, str(args.out_dir / "checkpoint.mpk"))
    else:
        # Create a new simulation & set the initial conditions.
        sim = Simulation(args.device, [200, 100], bgk(0.51))

        sim.fluid.vel[:, 40:60, 0] = 0.1
        sim.fluid.rho[:] += 0.1 * np.random.uniform(-1, 1, sim.fluid.rho.shape)

    # Run the simulation loop
    meta = IterInfo(250, 100)
    for i in run_sim(sim, meta, args.out_dir):
        # For each iteration generate an output
        dvydx = np.diff(sim.fluid.vel[..., 1], axis=0)[:, :-1]
        dvxdy = np.diff(sim.fluid.vel[..., 0], axis=1)[:-1, :]
        curl = dvydx - dvxdy
        plt.imshow(curl.T, cmap="RdBu")
        plt.show()

The :py:meth:`~boltzmann.simulation.parse_args` function loads the standard command line arguments 
for a simulation script and exposes them via a :py:class:`~boltzmann.simulation.SimulationArgs` object.
We can then use this to decide if we need to load the checkpoint or not create a fresh simulation.
If we are creating it afresh we will want to set some initial conditions.

Once we have a :py:class:`Simulation` object, comes the main loop.
To write this loop we use the :py:meth:`~boltzmann.simulation.run_sim` method,
which returns a ``Generator`` that yields the current output number on each iteration.
On each iteration of the ``for``-loop you generate your output (or do anything else you desire).
Once you have written your output the :py:meth:`~boltzmann.simulation.run_sim` generator will log information about the progress & performance, 
write the checkpoint (if requested) and then continue.

To tell :py:meth:`~boltzmann.simulation.run_sim` how often, and how many times, to yield we pass it an :py:class:`~boltzmann.simulation.IterInfo`.
This is a simple dataclass containing two pieces of information: the number of simulation time-steps per output, and how many outputs to generate.
The number of LBM iterations per output will depend on the physical system you are simulating and the temporal resolution you need.
For more info on this see :doc:`guides/units <units>`.

This approach is explicit and easy to understand but comes with two downsides:

.. note::

    We can control the frequency with which the checkpoints are written with the ``checkpoints`` argument to :py:meth:`~boltzmann.simulation.run_sim`.
    This is one of the fields of :py:class:`~boltzmann.simulation.SimulationArgs`; if you want to use it you must wire it into the :py:meth:`~boltzmann.simulation.run_sim` call.
    This goes for any extra arguments to :py:meth:`~boltzmann.simulation.run_sim` that appear in future.

.. note::
    
    With this approach we are responsible for loading from the checkpoint ourselves if requested by the user.
    The :py:meth:`~boltzmann.simulation.run_sim` function saves to a standard filename of ``checkpoint.mpk`` so we should load from there.

2. Callback based
-----------------

The second way to write such a script is to use the :py:class:`~boltzmann.simulation.SimulationScript` class.
Under the hood it does the same thing as the imperative approach, but it automatically handles wiring all command line arguments into the :py:meth:`~boltzmann.simulation.run_sim` call and loading the checkpoint (if requested).
This can be slightly cleaner but it can make things appear a bit more magical in certain situations.

Here's an example:

.. code-block:: python


    from boltzmann.simulation import IterInfo, SimulationScript
    
    # Simulation loop will run when the with-statement exists
    meta = IterInfo(250, 100)
    with (script := SimulationScript([200, 100], 1/0.51, meta)) as sim:

        @script.init
        def init():
            # Set the initial conditions, if needed.
            sim.fluid.vel[:, 40:60, 0] = 0.1
            sim.fluid.rho[:] += 0.1 * np.random.uniform(-1, 1, sim.fluid.rho.shape)

        @script.out
        def out(out_dir: Path, iter: int):
            # For each iteration generate an output
            dvydx = np.diff(sim.fluid.vel[..., 1], axis=0)[:, :-1]
            dvxdy = np.diff(sim.fluid.vel[..., 0], axis=1)[:-1, :]
            curl = dvydx - dvxdy
            plt.imshow(curl.T, cmap="RdBu")
            plt.show()

Note the use of decorators on functions which capture the surrounding scope. 
These functions are called only when needed; for example if we are resuming the `init` function will not get called.

.. note::

    The slightly magical use of the `walrus operator <https://docs.python.org/3/whatsnew/3.8.html>`_ 
    is because :py:class:`~boltzmann.simulation.SimulationScript`'s ``__enter__`` method returns a 
    :py:class:`~boltzmann.core.Simulation` object directly for you to work with, but we still need
    the :py:class:`~boltzmann.simulation.SimulationScript` to use the decorators which mark the 
    initialization and output methods. 


Running
-------

Whichever approach we take we can run the above scripts from the command line

.. code-block:: bash
    
    $ python my_sim.py --help
    usage: script.py [-h] [--device {gpu,cpu}] [--resume] [--out-dir OUT_DIR] [--checkpoints CHECKPOINTS]

    options:
    -h, --help            show this help message and exit
    --device {gpu,cpu}    The OpenCL device to run on
    --resume              Resume from the checkpoint.
    --out-dir OUT_DIR     Path to the directory where the checkpoint is stored.
    --checkpoints CHECKPOINTS
                            How often to write checkpoints.

    $ python my_sim.py --out-dir=./cool_results 

    2025-03-06 13:07:52,325 - INFO  - [logger] Output directory ........ ./cool_results 
    2025-03-06 13:07:52,325 - INFO  - [logger] Memory usage ....................... 104 MB
    2025-03-06 13:07:52,325 - INFO  - [logger] Iters / output .................... 1000 
    2025-03-06 13:07:52,325 - INFO  - [logger] Cells ............................. 2.0M 
    2025-03-06 13:07:56,408 - INFO  - [simulation] Batch 1: 0:00:02 1,311 MLUP/s, sim 1.5s, out 0.9s, eta 13:11:51
    2025-03-06 13:07:58,722 - INFO  - [simulation] Batch 2: 0:00:04 1,633 MLUP/s, sim 1.2s, out 1.1s, eta 13:11:48
    2025-03-06 13:08:00,566 - INFO  - [simulation] Batch 3: 0:00:06 1,599 MLUP/s, sim 1.3s, out 0.6s, eta 13:11:31
    2025-03-06 13:08:02,368 - INFO  - [simulation] Batch 4: 0:00:08 1,628 MLUP/s, sim 1.2s, out 0.6s, eta 13:11:22
    2025-03-06 13:08:04,349 - INFO  - [simulation] Batch 5: 0:00:10 1,627 MLUP/s, sim 1.2s, out 0.8s, eta 13:11:20
