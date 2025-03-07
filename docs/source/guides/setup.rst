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

- Turn cells into no-slip walls.  :py:attr:`CellFlags.WALL`
- Fix the velocity to the initial value (e.g. for in- and out-flow.) :py:attr:`CellFlags.FIXED_FLUID`
- Fix a scalar field's value (e.g. for a source of tracer.) :py:attr:`CellFlags.FIXED_SCALAR`

Initial Conditions
------------------

The initial conditions are set by updating the velocity and density arrays before running the simulation.
To set the initial values we access the :py:attr:`~Fluid.rho` and :py:attr:`~Fluid.vel` arrays via the :py:attr:`Simulation.fluid` member.

.. note::
    
    Before the first iteration the simulation will automatically use the macroscopic values to infer the equilibrium distributions for the microscopic velocities.

Body Force / Gravity
--------------------

A body force is a force which applies uniformly throughout the simulation domain.
It is set via the :py:meth:`Simulation.set_gravity` method, which accepts an acceleration *in lattice units*.

Tracers
-------

Tracers are scalar fields that are advected by the fluid.
We can add multiple tracers to a simulation.
Add tracers by calling the :py:meth:`Simulation.add_tracer` method with a name and the relevant parameters.
The method returns a reference to the :py:class:`Tracer`, which you can use to access the data.
The scalar value array is accessible via the :py:attr:`Tracer.val` attribute.

Checkpointing
-------------

To save the simulation state to disk call :py:meth:`~boltzmann.core.Simulation.write_checkpoint` on your :py:class:`~boltzmann.core.Simulation` object.
This will write a `MessagePack <https://msgpack.org/index.html>`_ file to disk at the specified path.
The saved state can then be loaded using the :py:meth:`~boltzmann.core.Simulation.load_checkpoint` static method.
The deserialized simulation has everything exactly as it was when it was saved, so is good to go.
