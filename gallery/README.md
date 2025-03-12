# Gallery

Some nice pictures of simulations produced with this package.
The code for all of these is included in the [examples directory (`/python/examples`)](/python/examples).
They were all run on my local desktop PC, which has an RTX 4060 Ti graphics card with 16GB of VRAM.

## Flow Past a Cylinder

![Flow Past a Cylinder](banner.png)

2D simulation with 27 million grid cells at a Reynold's number of 10,000.
The flow undergoes [vortex shedding](https://en.wikipedia.org/wiki/Vortex_shedding) and is somewhat turbulent. 

_Code at [`vortex_street/vortex_street.py`](/python/examples/vortex_street/vortex_street.py)._

## Colliding Jets

![Colliding jets with tracers](jets_2d.png)

An example simulation showing the use of tracers.
The simulation is of two jets colliding head-on. 
Each jet & its motion is highlighted by the tracers, here coloured yellow and purple for each jet.

_Code at [`jets_2d/jets.py`](/python/examples/jets_2d/jets.py)._

## Porous Media

<p>
    <div align="center" style="display: flex; flex-wrap: nowrap;">
        <image src="voronoi_velocity.png" width=48%px/>
        <image src="voronoi_tracer.png" width=48%px/>
    </div>
</p>

Shows the flow through a periodic 2D [packed-bed](https://en.wikipedia.org/wiki/Packed_bed) with procedurally generated particles.
The simulation has 2.3 million grid cells with a Reynold's number of ~250.

_Code at [`voronoi/voronoi.py`](/python/examples/voronoi/voronoi.py)_

## Colliding Jets 3D

<p>
    <div align="center" style="display: flex; flex-wrap: nowrap;">
        <image src="jets_3d_velocity.png" width=48%px/>
        <image src="jets_3d_qcriterion.png" width=48%px/>
    </div>
</p>

3D simulation showing two slightly off-center jets colliding.
The volumetric plots show the velocity magnitude (orange color-scheme) and the Q-criterion (green color-scheme).
The simulation has 64 million grid cells.

_Code at [`jets_3d/jets.py`](/python/examples/voronoi/voronoi.py)_

## Kelvin-Helmholtz Instability

<p>
    <div align="center" style="display: flex; flex-wrap: nowrap;">
        <image src="kelvin_helmholtz_tracer.png" width=48%px/>
        <image src="kelvin_helmholtz_vorticity.png" width=48%px/>
    </div>
</p>

Simulation showing the development of the [Kelvin-Helmholtz instability](https://en.wikipedia.org/wiki/Kelvin%E2%80%93Helmholtz_instability) at a shear boundary.
The top of the plot is moving to the right, while the bottom is moving to the left. 
This situation is unstable and slight perturbations grow and form large vortices.
This motion is highlighted by showing a tracer which is advected with the fluid.
The simulation has 32 million grid cells.

_Code at [`kelvin_helmholtz/kelvin_helmholtz.py`](/python/examples/kelvin_helmholtz/kelvin_helmholtz.py)_

## Boussinesq Approximation

<p  align="center">
    <div align="center" style="display: flex; flex-wrap: nowrap;">
        <a href="https://youtu.be/YyjomsE06RA">
            <image src="boussinesq_temp_1.png" width=31%px/>
            <image src="boussinesq_temp_2.png" width=31%px/>
            <image src="boussinesq_temp_3.png" width=31%px/>
        </a>
    </div>
    </br>
    <i>
    (Click images for a video.)
    </i>
</p>

Using the [Boussinesq approximation](https://en.wikipedia.org/wiki/Boussinesq_approximation_(buoyancy)) 
we can couple a tracer representing temperature to the fluid.
The fluid is initially at rest but a cold and heat source in the domain cause a convection current to establish itself.
The simulation has 4 million grid cells.

_Code at [`boussinesq/boussinesq.py`](/python/examples/boussinesq/boussinesq.py)_

## Rayleigh-Bénard Convection

<p  align="center">
    <div align="center" style="display: flex; flex-wrap: nowrap;">
        <a href="https://youtu.be/7ZdQ-CJM8yc">
            <image src="rayleigh_benard_temp_1.png" width=31%px/>
            <image src="rayleigh_benard_temp_2.png" width=31%px/>
            <image src="rayleigh_benard_temp_3.png" width=31%px/>
        </a>
    </div>
    </br>
    <i>
    (Click images for a video.)
    </i>
</p>

Using the same Boussinesq coupling we can set up a simulation of turbulent [Rayleigh–Bénard convection](https://en.wikipedia.org/wiki/Rayleigh%E2%80%93B%C3%A9nard_convection).
The whole bottom boundary is heated, while the top boundary is simultaneously cooled.
The fluid is initially completely at rest but the temperature differentials caused by the boundary conditions start the fluid convecting.
The simulation has 15 million cells.

_Code at [`rayleigh_benard/rayleigh_benard.py`](/python/examples/rayleigh_benard/rayleigh_benard.py)._