use ndarray::{Array1, Array2, ArrayView2, Zip};
use opencl3::memory::Buffer;
use rmp_serde::{Deserializer, Serializer};
use serde::{
    de::{DeserializeSeed, Visitor},
    Deserialize, Serialize,
};

use crate::{
    opencl::{
        AllocateCL, CtxDeserializer, Data, Data1d, Data1dDeserializer, Data2d, Data2dDeserializer,
        OpenCLCtx,
    },
    raster::Ix,
    velocities::VelocitySet,
};

pub trait MemUsage {
    fn size_bytes(&self) -> usize;
}

/// Long-lived container for the fluid Host and OpenCL device buffers.
/// This owns the simulation data and we expose a view to Python.
#[derive(Serialize)]
pub struct Fluid {
    pub f: Data2d<f32>,
    pub rho: Data1d<f32>,
    pub vel: Data2d<f32>,
    pub model: VelocitySet,
    pub omega: f32,
}

impl Fluid {
    pub fn new(opencl: &OpenCLCtx, counts: &Array1<Ix>, q: usize, omega: f32) -> Self {
        let n = counts.product() as usize;
        let d = counts.dim();

        Self {
            f: Data::new(opencl, [n, q], 0.0),
            rho: Data::new(opencl, [n], 1.0),
            vel: Data::new(opencl, [n, d], 0.0),
            model: VelocitySet::make(d, q),
            omega: omega,
        }
    }

    // Copy data from our host arrays into the OpenCL buffers.
    pub fn write_to_dev(&mut self, opencl: &OpenCLCtx) {
        let queue = &opencl.queue;
        let _write_f = self.f.enqueue_write(queue, "f");
        let _write_r = self.rho.enqueue_write(queue, "rho");
        let _write_v = self.vel.enqueue_write(queue, "vel");

        queue
            .finish()
            .expect("queue.finish() failed [Fluid::write_to_dev]");
    }

    /// Read the data back from the OpenCL buffers into our host arrays.
    pub fn read_to_host(&mut self, opencl: &OpenCLCtx) {
        let queue = &opencl.queue;
        let _read_f = self.f.enqueue_read(queue);
        let _read_r = self.rho.enqueue_read(queue);
        let _read_v = self.vel.enqueue_read(queue);

        queue
            .finish()
            .expect("queue.finish() failed [Fluid::read_to_host]");
    }

    pub fn equilibrate(&mut self) {
        // Rayonify this zip (though it's only called once at the beginning so doesn't need to be mega fast)
        Zip::from(&self.rho.host)
            .and(self.vel.host.rows())
            .and(self.f.host.rows_mut())
            .for_each(|&r, v, mut f| {
                f.assign(&self.model.feq(r, v));
            });
    }
}

impl MemUsage for Fluid {
    fn size_bytes(&self) -> usize {
        std::mem::size_of::<f32>() * (self.f.host.len() + self.rho.host.len() + self.vel.host.len())
    }
}

/// Long-lived container for scalar field Host and OpenCL device buffers.
/// This owns the simulation data and we expose a view to Python.
#[allow(non_snake_case)]
#[derive(Serialize)]
pub struct Scalar {
    pub g: Data2d<f32>,
    pub C: Data1d<f32>,
    pub model: VelocitySet,
    pub omega: f32,
}

#[allow(non_snake_case)]
impl Scalar {
    pub fn new(opencl: &OpenCLCtx, counts: &Array1<i32>, q: usize, omega: f32) -> Self {
        let n = counts.product() as usize;
        let d = counts.dim();

        Self {
            g: Data::new(opencl, [n, q], 0.0),
            C: Data::new(opencl, [n], 0.0),
            model: VelocitySet::make(d, q),
            omega: omega,
        }
    }

    // Copy data from our host arrays into the OpenCL buffers.
    pub fn write_to_dev(&mut self, opencl: &OpenCLCtx) {
        let queue = &opencl.queue;
        let _write_g = self.g.enqueue_write(queue, "q");
        let _write_C = self.C.enqueue_write(queue, "C");

        queue
            .finish()
            .expect("queue.finish() failed [Scalar::write_to_dev]");
    }

    /// Read the data back from the OpenCL buffers into our host arrays.
    pub fn read_to_host(&mut self, opencl: &OpenCLCtx) {
        let queue = &opencl.queue;
        let _read_g = self.g.enqueue_read(queue);
        let _read_C = self.C.enqueue_read(queue);

        queue
            .finish()
            .expect("queue.finish() failed [Scalar::read_to_host]");
    }

    pub fn equilibrate(&mut self, vel: ArrayView2<f32>) {
        // Rayonify this zip (though it's only called once at the beginning so doesn't need to be mega fast)
        Zip::from(&self.C.host)
            .and(vel.rows())
            .and(self.g.host.rows_mut())
            .for_each(|&C, v, mut g| {
                g.assign(&self.model.feq(C, v));
            });
    }
}

impl MemUsage for Scalar {
    fn size_bytes(&self) -> usize {
        std::mem::size_of::<f32>() * (self.g.host.len() + self.C.host.len())
    }
}

/// Because [`Data`] needs the [`OpenCLCtx`] to construct itself deserializing becomes a pain.
/// For this reason we have this "mirror" struct which we can deserialize then move out of to
/// create a raw [`Fluid`] object.
///
/// This is very infectious and quite gross. Unfortunately I don't see a good way to couple the
/// dev and host buffers together without having them live in the same struct.
#[derive(Deserialize)]
pub(crate) struct FluidDeserializer {
    f: Data2dDeserializer<f32>,
    rho: Data1dDeserializer<f32>,
    vel: Data2dDeserializer<f32>,
    model: VelocitySet,
    omega: f32,
}

impl CtxDeserializer for FluidDeserializer {
    type Target = Fluid;

    fn with_context(self, opencl: &OpenCLCtx) -> Self::Target {
        Self::Target {
            f: self.f.with_context(opencl),
            rho: self.rho.with_context(opencl),
            vel: self.vel.with_context(opencl),
            model: self.model,
            omega: self.omega,
        }
    }
}

/// See [`FluidDeserializer`]
#[derive(Deserialize)]
#[allow(non_snake_case)]
pub(crate) struct ScalarDeserializer {
    g: Data2dDeserializer<f32>,
    C: Data1dDeserializer<f32>,
    model: VelocitySet,
    omega: f32,
}

impl CtxDeserializer for ScalarDeserializer {
    type Target = Scalar;

    fn with_context(self, opencl: &OpenCLCtx) -> Self::Target {
        Self::Target {
            g: self.g.with_context(opencl),
            C: self.C.with_context(opencl),
            model: self.model,
            omega: self.omega,
        }
    }
}
