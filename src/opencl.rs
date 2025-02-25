use ndarray::{Array, Dimension, ShapeBuilder};
use numpy::{Ix1, Ix2, IxDyn};
use opencl3::{
    command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE},
    context::Context,
    device::{get_all_devices, Device, CL_DEVICE_TYPE_CPU, CL_DEVICE_TYPE_GPU},
    event::Event,
    kernel::Kernel,
    memory::{Buffer, CL_MEM_READ_WRITE},
    program::Program,
    types::{CL_BLOCKING, CL_NON_BLOCKING},
};
use serde::{Deserialize, Serialize};

const OPENCL_SRC: &str = include_str!("lib.cl");

pub enum DeviceType {
    CPU,
    GPU,
}

/// Long-lived OpenCL context.
pub struct OpenCLCtx {
    _device: Device,
    pub context: Context,
    pub queue: CommandQueue,
    pub d2q9_ns_kernel: Kernel,
    pub d2q5_ad_kernel: Kernel,
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
        let d2q9 = Kernel::create(&program, "update_d2q9_bgk").expect("Kernel::create failed D2Q9");
        let d2q5 = Kernel::create(&program, "update_d2q5_bgk").expect("Kernel::create failed D2Q5");

        Self {
            _device: device,
            context: context,
            queue: queue,
            d2q9_ns_kernel: d2q9,
            d2q5_ad_kernel: d2q5,
        }
    }
}

/// Long-lived container for host- and OpenCL device buffers.
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
    pub fn new<Sh>(opencl: &OpenCLCtx, shape: Sh, fill: T) -> Self
    where
        T: Clone,
        Sh: ShapeBuilder<Dim = D>,
    {
        Data::from_host(opencl, Array::from_elem(shape, fill))
    }

    /// Consume an [`Array`] and create a new [`Data`].
    pub fn from_host(opencl: &OpenCLCtx, host: Array<T, D>) -> Self {
        Self {
            dev: Data::make_buffer(&opencl.context, &host),
            host,
        }
    }

    /// Construct an OpenCL [`Buffer`] from the passed [`VectDView`].
    pub fn make_buffer(ctx: &Context, arr_host: &Array<T, D>) -> Buffer<T> {
        unsafe {
            Buffer::create(
                ctx,
                CL_MEM_READ_WRITE,
                arr_host.raw_dim().size(),
                std::ptr::null_mut(),
            )
            .expect("Buffer::create() failed")
        }
    }

    /// Write our host array to the OpenCL [`Buffer`].
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
                    CL_BLOCKING,
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
        Self::Target::from_host(opencl, self.host)
    }
}

pub type Data1d<T> = Data<T, Ix1>;
pub type Data2d<T> = Data<T, Ix2>;
pub type DataNd<T> = Data<T, IxDyn>;

pub type Data1dDeserializer<T> = DataDeserializer<T, Ix1>;
pub type Data2dDeserializer<T> = DataDeserializer<T, Ix2>;
pub type DataNdDeserializer<T> = DataDeserializer<T, IxDyn>;
