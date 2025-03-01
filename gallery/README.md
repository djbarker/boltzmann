# Gallery

Some nice pictures of simulations produced with this package.
The code for all of these is included in the [examples directory (`/python/examples`)](/python/examples).
They were all run on my local desktop PC, which has an RTX 4060 Ti graphics card.

## Flow Past a Cylinder

![Flow Past a Cylinder](banner.png)

2D simulation with 27 million grid cells at a Reynold's number of 10,000.
The flow undergoes [vortex shedding](https://en.wikipedia.org/wiki/Vortex_shedding) and is somewhat turbulent. 

_Code at [`vortex_street/vortex_street.py`](/python/examples/vortex_street/vortex_street.py)._

## Colliding Jets

![Colliding jets with tracers](jets.png)

An example simulation showing the use of tracers.
The simulation is of two jets colliding head-on. 
Each jet & its motion is highlighted by the tracers, here coloured yellow and purple for each jet.

_Code at [`jets/jets.py`](/python/examples/jets/jets.py)._

## Porous Media

<p>
    <div align="center" style="display: flex; flex-wrap: nowrap;">
        <image src="voronoi_velocity.png" height=300px/>
        <image src="voronoi_tracer.png" height=300px/>
    </div>
</p>

Shows the flow through a 2D [packed-bed](https://en.wikipedia.org/wiki/Packed_bed) with procedurally generated particles.
The simulation has 2.3 million grid cells with a Reynold's number of ~250.

_Code at [`voronoi/voronoi.py`](/python/examples/voronoi/voronoi.py)_