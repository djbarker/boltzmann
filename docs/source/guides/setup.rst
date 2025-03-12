Simulation Setup
================

This page is intended as a reference more than a guide but the sections are ordered in the way you would use them to set-up a simulation.

.. note::

    For a brief tutorial style introduction refer to the guides at :doc:`guides/basic basic` and :doc:`guides/units units`.


Array Indexing
--------------

All simulation data is exposed as multidimensional Numpy arrays. 
The indices are ordered as *x*, *y*, *z* and data is stored in row-major order.
For the velocity array the index of the last component matches this order.

Example
^^^^^^^

For a 2D simulation with grid size ``(nx, ny)``, 

- The velocity array (``sim.fluid.vel``) will have shape ``(nx, ny, 2)``.
- The density array (``sim.fluid.rho``) will have shape ``(nx, ny)``.
- The flags array (``sim.cells.flags``) will have shape ``(nx, ny)``.

Boundary Conditions
-------------------

Boundary conditions are configured by setting flags on the relevant cells.
To do this we update the relevant values under :py:attr:`Simulation.cells.flags`.
Permissible values are defined in :py:class:`boltzmann.core.CellFlags`.
Values can be combined. (Although not all combinations are very meaningful.)

The allows us to do things such as 

- Turn cells into no-slip walls - :py:attr:`~boltzmann.core.CellFlags.WALL`
- Fix the velocity to the initial value (e.g. for in- and out-flow) - :py:attr:`~boltzmann.core.CellFlags.FIXED_FLUID`
- Fix a scalar field's value (e.g. for a source of tracer) - :py:attr:`~boltzmann.core.CellFlags.FIXED_SCALAR`

Initial Conditions
------------------

The initial conditions are set by updating the velocity and density arrays before running the simulation.
To set the initial values we access the :py:attr:`~Fluid.rho` and :py:attr:`~Fluid.vel` arrays via the :py:attr:`~boltzmann.core.Simulation.fluid` member of the :py:class:`~boltzmann.core.Simulation`.

To aid in setting up the initial conditions the :py:class:`~boltzmann.units.Domain` class exposes position arrays in physical units.
See the documentation at :doc:`guides/units units` for more information.

.. note::
    
    Before the first iteration the simulation will automatically use the macroscopic values to infer the equilibrium distributions for the microscopic velocities.

Tracers
-------

Tracers are scalar fields that are advected by the fluid.
We can add multiple tracers to a simulation.
Add tracers by calling the :py:meth:`~boltzmann.core.Simulation.add_tracer` method with a name and the relevant parameters.
The method returns a reference to the :py:class:`~boltzmann.core.Scalar`, which you can use to access the data.
The scalar value array is accessible via the :py:attr:`~boltzmann.core.Scalar.val` attribute.

Body Force (Gravity)
--------------------

A body force is a force which applies uniformly throughout the simulation domain and is constant over time.
It is set via the :py:meth:`~boltzmann.core.Simulation.set_gravity` method, which accepts an acceleration *in lattice units*.

Fluid-Tracer Coupling
---------------------

The fluid can be coupled to the tracers via the `Boussinesq approximation <https://en.wikipedia.org/wiki/Boussinesq_approximation_(buoyancy)>`_.
This lets the fluid feel a buoyancy force proportional to the tracer concentration.
For example the tracer may represent the temperature or salinity of the fluid.
The exact form of the coupling is

.. math::

    \mathbf{a} = \alpha (C - C_0) \mathbf{g}

where :math:`\mathbf{a}` is the acceleration due to buoyancy, :math:`\alpha` is the coupling coefficient, :math:`C` is the tracer concentration, :math:`C_0` is the reference concentration and :math:`\mathbf{g}` is the gravitational acceleration.

To add a coupling to a tracer call :py:meth:`~boltzmann.core.Simulation.add_boussinesq_coupling` with the relevant parameters.


Checkpointing
-------------

To save the simulation state to disk call :py:meth:`~boltzmann.core.Simulation.write_checkpoint` on your :py:class:`~boltzmann.core.Simulation` object.
This will write a `MessagePack <https://msgpack.org/index.html>`_ file to disk at the specified path.
The saved state can then be loaded using the :py:meth:`~boltzmann.core.Simulation.load_checkpoint` static method.
The deserialized simulation has everything exactly as it was when it was saved, so is good to go.
