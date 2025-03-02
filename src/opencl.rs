use std::collections::HashMap;

use ndarray::{Array, Dimension, ShapeBuilder};
use numpy::{Ix1, Ix2, IxDyn};
use opencl3::{
    command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE},
    context::Context,
    device::{get_all_devices, Device, CL_DEVICE_TYPE_CPU, CL_DEVICE_TYPE_GPU},
    event::Event,
    kernel::Kernel,
    memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE},
    program::Program,
    types::{cl_mem_flags, CL_NON_BLOCKING},
};
use serde::{Deserialize, Serialize};

const OPENCL_SRC: &str = include_str!("lib.cl");

/// The type of device to use for the OpenCL computations.
pub enum DeviceType {
    CPU,
    GPU,
}

/// The implemented OpenCL kernels differ in whether they update the velocity or not.
/// [`EqnType::NavierStokes`] does, whereas [`EqnType::AdvectionDiffusion`] does not.
#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub enum EqnType {
    NavierStokes,
    AdvectionDiffusion,
}

/// Unique identifier for each kernel.
#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub struct KernelKey {
    pub d: usize,
    pub q: usize,
    pub eqn: EqnType,
}

impl KernelKey {
    pub fn new(d: usize, q: usize, eqn: EqnType) -> Self {
        Self { d, q, eqn }
    }
}

/// Long-lived OpenCL context.
pub struct OpenCLCtx {
    _device: Device,
    pub context: Context,
    pub queue: CommandQueue,
    pub kernels: HashMap<KernelKey, Kernel>,
}

impl OpenCLCtx {
    pub fn new(device_type: DeviceType) -> Self {
        let device_type = match device_type {
            DeviceType::CPU => CL_DEVICE_TYPE_CPU,
            DeviceType::GPU => CL_DEVICE_TYPE_GPU,
        };

        let device_id = *get_all_devices(device_type)
            .expect("error getting platform")
            .first()
            .expect("no device found in platform");
        let device = Device::new(device_id);
        let context = Context::from_device(&device).expect("Context::from_device failed");
        let queue = CommandQueue::create_default(&context, CL_QUEUE_PROFILING_ENABLE)
            .expect("CommandQueue::create_default failed");

        let program = Program::create_and_build_from_source(&context, OPENCL_SRC, "")
            .expect("Program::create_and_build_from_source failed");

        let keys = [
            (2, 9, EqnType::NavierStokes),
            (3, 27, EqnType::NavierStokes),
            (2, 5, EqnType::AdvectionDiffusion),
            (3, 7, EqnType::AdvectionDiffusion),
        ];

        let mut kernels = HashMap::new();
        for (d, q, e) in keys {
            let m = KernelKey::new(d, q, e);
            let k = Kernel::create(&program, format!("update_d{}q{}_bgk", d, q).as_str())
                .expect(format!("Kernel::create failed D{}Q{}", d, q).as_str());
            kernels.insert(m, k);
        }

        Self {
            _device: device,
            context: context,
            queue: queue,
            kernels: kernels,
        }
    }
}

/// Long-lived container for host- and OpenCL device buffers.
/// TODO: We could encore the read-write/read-only status in the type.
#[derive(Serialize)]
pub struct Data<T, D>
where
    D: Dimension,
{
    /// This is [`Option`] so we can deserialize and setup the buffers later.
    /// It also means we can change the [`OpenCLCtx`] without reallocating the host arrays.
    #[serde(skip)]
    pub dev: Buffer<T>,
    pub host: Array<T, D>,
}

impl<T, D: Dimension> Data<T, D> {
    /// Create an [`Array`] with the given shape, and turn it into a [`Data`].
    pub fn from_val<Sh>(opencl: &OpenCLCtx, shape: Sh, fill: T, flags: cl_mem_flags) -> Self
    where
        T: Clone,
        Sh: ShapeBuilder<Dim = D>,
    {
        Data::from_host(opencl, Array::from_elem(shape, fill), flags)
    }

    /// Consume an [`Array`] and create a new [`Data`].
    pub fn from_host(opencl: &OpenCLCtx, host: Array<T, D>, flags: cl_mem_flags) -> Self {
        Self {
            dev: Data::make_buffer(&opencl.context, &host, flags),
            host,
        }
    }

    /// Construct an OpenCL [`Buffer`] from the passed [`Array`].
    pub fn make_buffer(ctx: &Context, arr_host: &Array<T, D>, flags: cl_mem_flags) -> Buffer<T> {
        unsafe {
            Buffer::create(ctx, flags, arr_host.raw_dim().size(), std::ptr::null_mut())
                .expect("Buffer::create() failed")
        }
    }

    /// Create an [`Array`] with the given shape, and turn it into a read-writable [`Data`].
    pub fn from_val_rw<Sh>(opencl: &OpenCLCtx, shape: Sh, fill: T) -> Self
    where
        T: Clone,
        Sh: ShapeBuilder<Dim = D>,
    {
        Data::from_val(opencl, shape, fill, CL_MEM_READ_WRITE)
    }

    /// Create an [`Array`] with the given shape, and turn it into a read-writable [`Data`].
    pub fn from_val_ro<Sh>(opencl: &OpenCLCtx, shape: Sh, fill: T) -> Self
    where
        T: Clone,
        Sh: ShapeBuilder<Dim = D>,
    {
        Data::from_val(opencl, shape, fill, CL_MEM_READ_ONLY)
    }

    /// Consume an [`Array`] and create a new read-write [`Data`].
    pub fn from_host_rw(opencl: &OpenCLCtx, host: Array<T, D>) -> Self {
        Self::from_host(opencl, host, CL_MEM_READ_WRITE)
    }

    /// Consume an [`Array`] and create a new read-only [`Data`].
    pub fn from_host_ro(opencl: &OpenCLCtx, host: Array<T, D>) -> Self {
        Self::from_host(opencl, host, CL_MEM_READ_ONLY)
    }

    /// Write our host array to the OpenCL [`Buffer`].
    ///
    /// NOTE: This can look confusing when called on a buffer created with `CL_MEM_READ_ONLY`,
    ///       but the read-only refers to when it's on the device.
    ///       We still need to copy data over to the device.
    pub fn enqueue_write(&mut self, queue: &CommandQueue, msg: &str) -> Event {
        unsafe {
            queue
                .enqueue_write_buffer(
                    &mut self.dev,
                    CL_NON_BLOCKING,
                    0,
                    self.host.as_slice().unwrap(),
                    &[], // ignore events ... why?
                )
                .expect(format!("enqueue_write_buffer failed {}", msg).as_str())
        }
    }

    // Read back from OpenCL device buffers into our host arrays.
    pub fn enqueue_read(&mut self, queue: &CommandQueue) -> Event {
        assert_eq!(self.host.len(), self.host.as_slice().unwrap().len());
        unsafe {
            queue
                .enqueue_read_buffer(
                    &self.dev,
                    CL_NON_BLOCKING,
                    0,
                    self.host.as_slice_mut().unwrap(),
                    &[], // ignore events ... why?
                )
                .expect("enqueue_read_buffer failed")
        }
    }
}

/// Implemented by "deserializer" structs to create the full type with allocated OpenCL buffers.
pub trait CtxDeserializer {
    type Target;

    /// Consume the object to create the full type with allocated OpenCL buffers.
    fn with_context(self, opencl: &OpenCLCtx) -> Self::Target;
}

#[derive(Deserialize)]
pub struct DataDeserializer<T, D>
where
    D: Dimension,
{
    host: Array<T, D>,
}

impl<T, D> CtxDeserializer for DataDeserializer<T, D>
where
    D: Dimension,
{
    type Target = Data<T, D>;

    fn with_context(self, opencl: &OpenCLCtx) -> Self::Target {
        Self::Target::from_host_rw(opencl, self.host)
    }
}

pub type Data1d<T> = Data<T, Ix1>;
pub type Data2d<T> = Data<T, Ix2>;
pub type DataNd<T> = Data<T, IxDyn>;

pub type Data1dDeserializer<T> = DataDeserializer<T, Ix1>;
pub type Data2dDeserializer<T> = DataDeserializer<T, Ix2>;
pub type DataNdDeserializer<T> = DataDeserializer<T, IxDyn>;
