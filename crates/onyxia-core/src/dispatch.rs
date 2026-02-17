//! Dispatch-based execution types.
//!
//! These types form the core of the new execution model where operators
//! dispatch their own GPU work at runtime, with full knowledge of input
//! shapes and data.

use crate::Result;
use crate::plan::ModelMetadata;
use crate::types::DataType;
use std::collections::HashMap;
use std::sync::Arc;

/// A fully materialized GPU tensor with known shape and data.
///
/// `RuntimeTensor` is cheap to clone (the GPU buffer is `Arc`-shared),
/// so a tensor consumed by multiple downstream operations is shared,
/// not copied.
#[derive(Debug, Clone)]
pub struct RuntimeTensor {
    /// GPU buffer containing the tensor data.
    pub buffer: Arc<wgpu::Buffer>,

    /// Concrete shape dimensions (always fully known at runtime).
    pub shape: Vec<usize>,

    /// Element data type.
    pub dtype: DataType,

    /// Total buffer size in bytes.
    pub size_bytes: usize,
}

/// Runtime context for dispatching GPU compute work.
///
/// Passed to `OpDispatch::dispatch()` by the runtime. Provides access
/// to the GPU device/queue and caches compute pipelines for reuse
/// across dispatch calls.
pub struct DispatchCtx {
    /// GPU device for resource creation.
    pub device: Arc<wgpu::Device>,

    /// Command queue for GPU submissions.
    pub queue: Arc<wgpu::Queue>,

    /// Pipeline cache: maps naga module pointer to (pipeline, bind_group_layout).
    /// Pipelines are created lazily on first use and reused across dispatches.
    pipeline_cache: HashMap<PipelineCacheKey, CachedPipeline>,
}

/// Key for pipeline cache lookups.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct PipelineCacheKey {
    /// Label used to identify the shader (unique per operator instance).
    label: String,
}

/// A cached compute pipeline and its bind group layout.
struct CachedPipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl DispatchCtx {
    /// Create a new dispatch context.
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        Self {
            device,
            queue,
            pipeline_cache: HashMap::new(),
        }
    }

    /// Allocate a new GPU buffer for an output tensor.
    pub fn create_output_tensor(&self, shape: &[usize], dtype: DataType) -> Result<RuntimeTensor> {
        let num_elements: usize = shape.iter().product();
        let size_bytes = num_elements * dtype.size();
        // wgpu requires buffers to be at least 4 bytes
        let buffer_size = size_bytes.max(4) as u64;

        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(RuntimeTensor {
            buffer: Arc::new(buffer),
            shape: shape.to_vec(),
            dtype,
            size_bytes,
        })
    }

    /// Upload data from CPU to a new GPU buffer, returning a RuntimeTensor.
    pub fn upload_tensor(
        &self,
        data: &[u8],
        shape: &[usize],
        dtype: DataType,
    ) -> Result<RuntimeTensor> {
        let num_elements: usize = shape.iter().product();
        let size_bytes = num_elements * dtype.size();
        assert_eq!(data.len(), size_bytes, "Data length mismatch");
        let buffer_size = size_bytes.max(4) as u64;

        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        self.queue.write_buffer(&buffer, 0, data);

        Ok(RuntimeTensor {
            buffer: Arc::new(buffer),
            shape: shape.to_vec(),
            dtype,
            size_bytes,
        })
    }

    /// Get or create a compute pipeline from a pre-compiled naga module.
    ///
    /// The pipeline is cached by label — subsequent calls with the same
    /// label return the cached pipeline without recompilation.
    pub fn get_or_create_pipeline(
        &mut self,
        label: &str,
        module: &naga::Module,
        entry_point: &str,
    ) -> Result<(&wgpu::ComputePipeline, &wgpu::BindGroupLayout)> {
        let key = PipelineCacheKey {
            label: label.to_string(),
        };

        if !self.pipeline_cache.contains_key(&key) {
            let shader_module = self
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(label),
                    source: wgpu::ShaderSource::Naga(std::borrow::Cow::Owned(module.clone())),
                });

            let bind_group_layout =
                self.device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some(&format!("{label}_layout")),
                        entries: &[], // Will be populated per-operator
                    });

            let pipeline_layout =
                self.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some(&format!("{label}_pipeline_layout")),
                        bind_group_layouts: &[&bind_group_layout],
                        immediate_size: 0,
                    });

            let pipeline = self
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(label),
                    layout: Some(&pipeline_layout),
                    module: &shader_module,
                    entry_point: Some(entry_point),
                    compilation_options: Default::default(),
                    cache: None,
                });

            self.pipeline_cache.insert(
                key.clone(),
                CachedPipeline {
                    pipeline,
                    bind_group_layout,
                },
            );
        }

        let cached = self.pipeline_cache.get(&key).unwrap();
        Ok((&cached.pipeline, &cached.bind_group_layout))
    }
}

/// Trait for runtime operation dispatch.
///
/// Each `OpDispatch` implementation is a self-contained operation that
/// knows how to execute itself on the GPU given concrete input tensors.
/// The dispatch object captures everything it needs at compile time
/// (pre-compiled shaders, attributes, etc.) and computes shapes,
/// workgroup dimensions, and immediates at runtime from actual inputs.
///
/// # Example
///
/// ```ignore
/// struct AddDispatch {
///     module: naga::Module,
/// }
///
/// impl OpDispatch for AddDispatch {
///     fn dispatch(
///         &self,
///         inputs: Vec<RuntimeTensor>,
///         ctx: &mut DispatchCtx,
///     ) -> Result<Vec<RuntimeTensor>> {
///         let a = &inputs[0];
///         let b = &inputs[1];
///         let output_shape = broadcast_shape(&a.shape, &b.shape)?;
///         let output = ctx.create_output_tensor(&output_shape, a.dtype)?;
///         // ... dispatch compute shader ...
///         Ok(vec![output])
///     }
/// }
/// ```
pub trait OpDispatch: Send + Sync {
    /// Execute this operation on the GPU.
    ///
    /// # Arguments
    ///
    /// * `inputs` — Input tensors collected from previous operations.
    ///   Shapes, dtypes, and data are all concrete and fully known.
    /// * `ctx` — GPU dispatch context for pipeline creation, buffer
    ///   allocation, and shader dispatch.
    ///
    /// # Returns
    ///
    /// Output tensors produced by this operation, with concrete shapes
    /// and data resident on the GPU.
    fn dispatch(
        &self,
        inputs: Vec<RuntimeTensor>,
        ctx: &mut DispatchCtx,
    ) -> Result<Vec<RuntimeTensor>>;
}

/// A compiled model ready for dispatch-based execution.
///
/// Contains a sequence of dispatch entries with register routing,
/// weight data, and input/output mappings.
pub struct CompiledModel {
    /// Operations to dispatch in order.
    pub entries: Vec<DispatchEntry>,

    /// Total number of register slots for tensor routing.
    pub num_registers: usize,

    /// Mapping from ONNX input names to register indices.
    pub input_registers: Vec<(String, usize)>,

    /// Mapping from ONNX output names to register indices.
    pub output_registers: Vec<(String, usize)>,

    /// Weight data to upload to GPU at load time.
    pub weight_registers: Vec<WeightRegister>,

    /// Model metadata.
    pub metadata: ModelMetadata,
}

/// A single operation in the dispatch sequence.
pub struct DispatchEntry {
    /// The dispatch implementation.
    pub op: Box<dyn OpDispatch>,

    /// Register indices to read inputs from.
    pub input_regs: Vec<usize>,

    /// Register indices to write outputs to.
    pub output_regs: Vec<usize>,

    /// Operation name for debugging/profiling.
    pub name: String,
}

/// Weight data to upload to a register at model load time.
pub struct WeightRegister {
    /// Register index to store the weight tensor in.
    pub register: usize,

    /// Raw weight bytes.
    pub data: Vec<u8>,

    /// Weight shape.
    pub shape: Vec<usize>,

    /// Weight data type.
    pub dtype: DataType,
}
