Parameters and Units
====================

The examples shown in previous sections have all used the raw 'simulation' (aka 'lattice') units.
To actually be useful for us we need two related things

1. Simulation parameters which correspond to our physical system.
2. A map between the simulation units and physical units.

The :py:mod:`boltzmann.core` and :py:mod:`boltzmann.units` modules provide utilities to help us with respective steps.

1. Parameter Choice
-------------------

For a detailed write-up of how to choose the simulation parameters such that they correspond to our physical system and the simulation remains stable, see `Lattice Boltzmann Parameterization <https://dbarker.uk/lbm_parameterization/>`_.
In short, though, we need to choose the collision timescale in LBM steps (``tau``), the velocity in lattice units (``u``), and the cell counts (``L``) such that the Reynold's number matches that of our physical system.

Given a full parameterization you can use the function :py:meth:`~boltzmann.core.check_lbm_params` to check if they are likely to result in a stable simulation.
If not it will raise a ``ValueError`` telling you which parameter you should consider changing.
This is not 100% foolproof; sometimes simulations it thinks are stable may still evolve into states that breach the stability limits, 
and those it thinks are likely to become unstable may remain stable for the duration of the simulation.
This is mostly due to some freedom in choosing the characteristic length-scale of the system.

Usually you know the Reynold's number and the number of cells in the characteristic length, rather than ``tau`` and ``u``. 
In which case the function :py:meth:`~boltzmann.core.calc_lbm_params_lu` will calculate suitable values for the collision timescale (``tau``) and the velocity (``u``) for you.
These values are chosen to maximize the speed of the simulation (rather than accuracy).
If you wish to prioritize accuracy at the cost of needing more time-steps you can manually pass a value of ``tau`` to override the default choice. 

Here is an example of calculating the LBM parameters for a physical system:

.. code-block:: python

    from boltzmann.core import Simulation, calc_lbm_params_lu, calc_lbm_params_si

    dx = 0.001 # Grid spacing [m]
    L_si = 1.0  # Characteristic length-scale [m]
    u_si = 1.0  # Characteristic velocity [m/s]
    nu_si = 1.0e-6  # Kinematic viscosity [m^2/s]

    Re = u_si * L_si / nu_si  # => Reynold's number
    L_lu = int(L_si / dx)  # Characteristic length-scale [cells]

    # Calculate collision timescale and characteristic velocity in lattice units:
    (tau, u) = calc_lbm_params_lu(Re, L_lu)

    # Now use these to create the simulation & set the initial conditions.
    sim = Simulation("cpu", [L_lu, L_lu], 1 / tau1)
    sim.fluid.vel[:, :, 0] = u1
    
.. note::

    The length-scale used in the parameter calculations is not the whole system size but a characteristic length-scale.
    It should match the length-scale used in the Reynold's number calculation. 
    Therefore if you are only calculating ``Re`` to pass it into :py:meth:`~boltzmann.core.calc_lbm_params_lu`
    you can use the alternative function :py:meth:`~boltzmann.core.calc_lbm_params_si`.
    This performs the calculations for you which ensures the length-scales used are consistent.

    .. code-block:: python
        
        (tau1, u1) = calc_lbm_params_lu(Re, L_lu)
        (tau2, u2) = calc_lbm_params_si(dx, u_si, L_si, nu_si)

        assert abs(tau1 - tau2) < 1e-6
        assert abs(u1 - u2) < 1e-6


2. Unit Conversion
------------------

In the above example we set our initial conditions in lattice units directly.
It would be nicer if we could set them in our physical units then convert them to lattice units latter.
Further, in order to plot the results we most likely want to convert back to physical units.
This ability is what the :py:class:`~boltzmann.units.Scales` class gives us.

It simply contains the conversion factors between the lattice units and the physical units for time, length and mass (the last of which is usually not relevant).
Using these it lets us convert dimensional quantities such as distances, velocities and accelerations between the two unit systems easily.

Here is an example of using the :py:class:`~boltzmann.units.Scales` class:

.. code-block:: python

    from boltzmann.units import Scales

    # ...

    scales = Scales(dx, dt)

    # Set initial conditions in physical units.
    sim.fluid.vel[:, :, 0] = u_si

    # Then convert to lattice units before running.
    sim.fluid.vel[:] = scales.velocity.to_lattice_units(sim.fluid.vel)

    # ...

    # If needed convert back, to plot in physical units.
    vel_mag_lu = np.sum(sim.fluid.vel**2, axis=-1)
    vel_mag_si = scales.velocity.to_physical_units(vel_mag_lu)

Behind the scenes the :py:class:`~boltzmann.units.Scales` class is used to build :py:class:`~boltzmann.units.UnitConverter` objects which are responsible for converting a specific dimensional quantity, e.g. distance, vorticity, density, etc.
For the most common cases of distance, velocity and acceleration the :py:class:`~boltzmann.units.Scales` class has convenience properties to get the :py:class:`~boltzmann.units.UnitConverter`, as seen in the example.
For arbitrary dimensional quantities you can pass the relevant powers of ``T``, ``L`` and ``M`` to the :py:meth:`~boltzmann.units.Scales.converter` method to construct a suitable :py:class:`~boltzmann.units.UnitConverter`.


The Domain Class
================

To make setting initial- and boundary conditions easy we use the :py:class:`~boltzmann.units.Domain` class.
This exposes

- :py:attr:`~boltzmann.units.Domain.x`, :py:attr:`~boltzmann.units.Domain.y`, :py:attr:`~boltzmann.units.Domain.z` - 1d arrays of cell positions for each dimension.
- :py:meth:`~boltzmann.units.Domain.meshgrid` - Tuple of multidimensional arrays of cell positions throughout the domain.

The :py:meth:`~boltzmann.units.Domain.meshgrid` method wraps a call to :py:func:`numpy.meshgrid`.
If you are unfamiliar with this function I suggest reading the Numpy documentation.

.. note::

    The positions are are the cell centres, not the cell edges.
    They are measured in physical units, not lattice units.

Example
^^^^^^^

Here is an example of using the :py:class:`~boltzmann.units.Domain` class to set some initial conditions
where velocity & density depend on the position in the domain:

.. code-block:: python

    import numpy as np

    from boltzmann.units import Domain

    # Construct the domain from the physical dimensions of the system.
    domain = Domain.make(lower=[-1, -1], upper=[1, 1], dx=0.01)

    # ...

    # Set the x-velocity to be sinusoidally varying along the x-axis.
    # NOTE: The extra axis is required for broadcasting to the 2D velocity array.
    sim.fluid.vel[:, :, 0] = np.sin(2 * np.pi * domain.x)[:, None]    

    # Set the density to be a Gaussian centred at the origin.
    XX, YY = domain.meshgrid()
    sim.fluid.density[:] = np.exp(-(XX**2 + YY**2))


To construct a :py:class:`~boltzmann.units.Domain` we can pass various combinations of arguments.
For full details refer to the class documentation.
