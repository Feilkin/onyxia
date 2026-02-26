//! Operator benchmarks for the Onyxia runtime.
//!
//! This benchmark suite measures the runtime performance of operators executing
//! on GPU. The benchmarks focus on:
//!
//! - **Unary operators** (6): Neg, Sqrt, Cos, Sin, Tanh, Gelu
//! - **Binary operators** (5): Add, Mul, Div, Sub, Pow
//! - **Comparison operators** (5): Equal, Greater, Less, GreaterOrEqual, LessOrEqual
//! - **Matrix operations** (1): MatMul with various sizes
//! - **Reduction operators** (2): ReduceMean, ReduceSum
//! - **Activation operators** (1): Softmax
//! - **Shape manipulation** (2): Transpose, Reshape
//! - **Type conversion** (1): Cast
//! - **Conditional operators** (1): Where
//!
//! Each benchmark creates a graph with 50-100 sequential operations of the same type
//! to amortize compile-time overhead and measure pure runtime performance across
//! different input sizes. Benchmarks use parametrized input sizes to identify
//! performance characteristics across different workload scales.
//!
//! Run benchmarks with: `cargo bench --bench operator_benchmarks`

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use pollster::block_on;
use std::collections::HashMap;

use onyxia_compiler::compile;
use onyxia_onnx::{AttributeValue, DataType, Graph, Node, TensorInfo, TensorKind, TensorShape};
use onyxia_operators::core_operator_registry;
use onyxia_runtime::{DispatchExecutor, Runtime, Tensor};

// ============================================================================
// Graph builders
// ============================================================================

fn unary_op_graph(op_type: &str, count: usize, io_size: usize) -> Graph {
    let mut graph = Graph::new();
    let mut previous_output = "input".to_string();

    graph.add_tensor(TensorInfo {
        name: previous_output.clone(),
        shape: TensorShape::Static(vec![io_size]),
        dtype: DataType::F32,
        kind: TensorKind::Input,
        initializer: None,
    });
    graph.inputs = vec![previous_output.clone()];

    for i in 0..count {
        let output = format!("op-{i}-output");
        graph.add_tensor(TensorInfo {
            name: output.clone(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![io_size]),
            kind: if i == count - 1 {
                TensorKind::Output
            } else {
                TensorKind::Intermediate
            },
            initializer: None,
        });

        graph.add_node(
            Node::new(op_type)
                .with_inputs(vec![previous_output])
                .with_outputs(vec![output.clone()]),
        );

        previous_output = output;
    }

    graph.outputs = vec![previous_output];
    graph
}

fn binary_op_graph(op_type: &str, count: usize, io_size: usize) -> Graph {
    let mut graph = Graph::new();
    let input1 = "input1".to_string();
    let input2 = "input2".to_string();

    // Add input tensors
    for input in [&input1, &input2] {
        graph.add_tensor(TensorInfo {
            name: input.clone(),
            shape: TensorShape::Static(vec![io_size]),
            dtype: DataType::F32,
            kind: TensorKind::Input,
            initializer: None,
        });
    }
    graph.inputs = vec![input1.clone(), input2.clone()];

    let mut previous_output = input1.clone();

    for i in 0..count {
        let output = format!("op-{i}-output");
        graph.add_tensor(TensorInfo {
            name: output.clone(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![io_size]),
            kind: if i == count - 1 {
                TensorKind::Output
            } else {
                TensorKind::Intermediate
            },
            initializer: None,
        });

        graph.add_node(
            Node::new(op_type)
                .with_inputs(vec![previous_output, input2.clone()])
                .with_outputs(vec![output.clone()]),
        );

        previous_output = output;
    }

    graph.outputs = vec![previous_output];
    graph
}

fn matmul_graph(count: usize, m: usize, k: usize, n: usize) -> Graph {
    let mut graph = Graph::new();
    let input_a = "input_a".to_string();
    let input_b = "input_b".to_string();

    graph.add_tensor(TensorInfo {
        name: input_a.clone(),
        shape: TensorShape::Static(vec![m, k]),
        dtype: DataType::F32,
        kind: TensorKind::Input,
        initializer: None,
    });

    graph.add_tensor(TensorInfo {
        name: input_b.clone(),
        shape: TensorShape::Static(vec![k, n]),
        dtype: DataType::F32,
        kind: TensorKind::Input,
        initializer: None,
    });

    graph.inputs = vec![input_a.clone(), input_b.clone()];

    let mut previous_output = input_a.clone();

    for i in 0..count {
        let output = format!("matmul-{i}-output");
        let out_shape = if i == 0 { vec![m, n] } else { vec![m, k] };

        graph.add_tensor(TensorInfo {
            name: output.clone(),
            dtype: DataType::F32,
            shape: TensorShape::Static(out_shape),
            kind: if i == count - 1 {
                TensorKind::Output
            } else {
                TensorKind::Intermediate
            },
            initializer: None,
        });

        let inputs = if i == 0 {
            vec![previous_output, input_b.clone()]
        } else {
            vec![previous_output, input_a.clone()]
        };

        graph.add_node(
            Node::new("MatMul")
                .with_inputs(inputs)
                .with_outputs(vec![output.clone()]),
        );

        previous_output = output;
    }

    graph.outputs = vec![previous_output];
    graph
}

fn reduction_graph(op_type: &str, count: usize, shape: Vec<usize>, axes: Vec<i64>) -> Graph {
    let mut graph = Graph::new();
    let mut previous_output = "input".to_string();

    graph.add_tensor(TensorInfo {
        name: previous_output.clone(),
        shape: TensorShape::Static(shape.clone()),
        dtype: DataType::F32,
        kind: TensorKind::Input,
        initializer: None,
    });
    graph.inputs = vec![previous_output.clone()];

    // Compute reduced shape
    let mut reduced_shape = shape.clone();
    for &axis in &axes {
        let axis_idx = if axis < 0 {
            (shape.len() as i64 + axis) as usize
        } else {
            axis as usize
        };
        reduced_shape[axis_idx] = 1;
    }

    for i in 0..count {
        let output = format!("op-{i}-output");
        graph.add_tensor(TensorInfo {
            name: output.clone(),
            dtype: DataType::F32,
            shape: TensorShape::Static(reduced_shape.clone()),
            kind: if i == count - 1 {
                TensorKind::Output
            } else {
                TensorKind::Intermediate
            },
            initializer: None,
        });

        let mut attrs = HashMap::new();
        attrs.insert("axes".to_string(), AttributeValue::Ints(axes.clone()));

        graph.add_node(
            Node::new(op_type)
                .with_inputs(vec![previous_output])
                .with_outputs(vec![output.clone()])
                .with_attributes(attrs),
        );

        previous_output = output;
    }

    graph.outputs = vec![previous_output];
    graph
}

fn softmax_graph(count: usize, shape: Vec<usize>, axis: i64) -> Graph {
    let mut graph = Graph::new();
    let mut previous_output = "input".to_string();

    graph.add_tensor(TensorInfo {
        name: previous_output.clone(),
        shape: TensorShape::Static(shape.clone()),
        dtype: DataType::F32,
        kind: TensorKind::Input,
        initializer: None,
    });
    graph.inputs = vec![previous_output.clone()];

    for i in 0..count {
        let output = format!("op-{i}-output");
        graph.add_tensor(TensorInfo {
            name: output.clone(),
            dtype: DataType::F32,
            shape: TensorShape::Static(shape.clone()),
            kind: if i == count - 1 {
                TensorKind::Output
            } else {
                TensorKind::Intermediate
            },
            initializer: None,
        });

        let mut attrs = HashMap::new();
        attrs.insert("axis".to_string(), AttributeValue::Int(axis));

        graph.add_node(
            Node::new("Softmax")
                .with_inputs(vec![previous_output])
                .with_outputs(vec![output.clone()])
                .with_attributes(attrs),
        );

        previous_output = output;
    }

    graph.outputs = vec![previous_output];
    graph
}

fn transpose_graph(count: usize, shape: Vec<usize>, perm: Vec<i64>) -> Graph {
    let mut graph = Graph::new();
    let mut previous_output = "input".to_string();

    graph.add_tensor(TensorInfo {
        name: previous_output.clone(),
        shape: TensorShape::Static(shape.clone()),
        dtype: DataType::F32,
        kind: TensorKind::Input,
        initializer: None,
    });
    graph.inputs = vec![previous_output.clone()];

    // Compute transposed shape
    let transposed_shape: Vec<usize> = perm.iter().map(|&i| shape[i as usize]).collect();

    for i in 0..count {
        let output = format!("op-{i}-output");
        let out_shape = if i % 2 == 0 {
            transposed_shape.clone()
        } else {
            shape.clone()
        };

        graph.add_tensor(TensorInfo {
            name: output.clone(),
            dtype: DataType::F32,
            shape: TensorShape::Static(out_shape),
            kind: if i == count - 1 {
                TensorKind::Output
            } else {
                TensorKind::Intermediate
            },
            initializer: None,
        });

        let perm_attr = if i % 2 == 0 {
            perm.clone()
        } else {
            // Inverse permutation
            let mut inv = vec![0; perm.len()];
            for (idx, &p) in perm.iter().enumerate() {
                inv[p as usize] = idx as i64;
            }
            inv
        };

        let mut attrs = HashMap::new();
        attrs.insert("perm".to_string(), AttributeValue::Ints(perm_attr));

        graph.add_node(
            Node::new("Transpose")
                .with_inputs(vec![previous_output])
                .with_outputs(vec![output.clone()])
                .with_attributes(attrs),
        );

        previous_output = output;
    }

    graph.outputs = vec![previous_output];
    graph
}

fn cast_graph(count: usize, io_size: usize) -> Graph {
    let mut graph = Graph::new();
    let mut previous_output = "input".to_string();

    graph.add_tensor(TensorInfo {
        name: previous_output.clone(),
        shape: TensorShape::Static(vec![io_size]),
        dtype: DataType::F32,
        kind: TensorKind::Input,
        initializer: None,
    });
    graph.inputs = vec![previous_output.clone()];

    for i in 0..count {
        let output = format!("op-{i}-output");
        // Alternate between F32 (ONNX code 1) and I32 (ONNX code 6)
        let (out_dtype, onnx_code) = if i % 2 == 0 {
            (DataType::I32, 6i64)
        } else {
            (DataType::F32, 1i64)
        };

        graph.add_tensor(TensorInfo {
            name: output.clone(),
            dtype: out_dtype,
            shape: TensorShape::Static(vec![io_size]),
            kind: if i == count - 1 {
                TensorKind::Output
            } else {
                TensorKind::Intermediate
            },
            initializer: None,
        });

        let mut attrs = HashMap::new();
        attrs.insert("to".to_string(), AttributeValue::Int(onnx_code));

        graph.add_node(
            Node::new("Cast")
                .with_inputs(vec![previous_output])
                .with_outputs(vec![output.clone()])
                .with_attributes(attrs),
        );

        previous_output = output;
    }

    graph.outputs = vec![previous_output];
    graph
}

fn where_graph(count: usize, io_size: usize) -> Graph {
    let mut graph = Graph::new();
    let cond = "cond".to_string();
    let x = "x".to_string();
    let y = "y".to_string();

    // Add input tensors
    graph.add_tensor(TensorInfo {
        name: cond.clone(),
        shape: TensorShape::Static(vec![io_size]),
        dtype: DataType::Bool,
        kind: TensorKind::Input,
        initializer: None,
    });

    for input in [&x, &y] {
        graph.add_tensor(TensorInfo {
            name: input.clone(),
            shape: TensorShape::Static(vec![io_size]),
            dtype: DataType::F32,
            kind: TensorKind::Input,
            initializer: None,
        });
    }

    graph.inputs = vec![cond.clone(), x.clone(), y.clone()];

    let mut previous_output = x.clone();

    for i in 0..count {
        let output = format!("op-{i}-output");
        graph.add_tensor(TensorInfo {
            name: output.clone(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![io_size]),
            kind: if i == count - 1 {
                TensorKind::Output
            } else {
                TensorKind::Intermediate
            },
            initializer: None,
        });

        graph.add_node(
            Node::new("Where")
                .with_inputs(vec![cond.clone(), previous_output, y.clone()])
                .with_outputs(vec![output.clone()]),
        );

        previous_output = output;
    }

    graph.outputs = vec![previous_output];
    graph
}

fn reshape_graph(count: usize, base_shape: Vec<usize>) -> Graph {
    let mut graph = Graph::new();
    let mut previous_output = "input".to_string();

    graph.add_tensor(TensorInfo {
        name: previous_output.clone(),
        shape: TensorShape::Static(base_shape.clone()),
        dtype: DataType::F32,
        kind: TensorKind::Input,
        initializer: None,
    });
    graph.inputs = vec![previous_output.clone()];

    let total_size: usize = base_shape.iter().product();
    // Alternate between flattened and original shape
    let alt_shape = vec![total_size];

    for i in 0..count {
        let output = format!("op-{i}-output");
        let out_shape = if i % 2 == 0 {
            alt_shape.clone()
        } else {
            base_shape.clone()
        };

        graph.add_tensor(TensorInfo {
            name: output.clone(),
            dtype: DataType::F32,
            shape: TensorShape::Static(out_shape.clone()),
            kind: if i == count - 1 {
                TensorKind::Output
            } else {
                TensorKind::Intermediate
            },
            initializer: None,
        });

        // Create shape tensor as input
        let shape_name = format!("shape-{i}");
        let shape_i64: Vec<i64> = out_shape.iter().map(|&x| x as i64).collect();
        let shape_bytes = bytemuck::cast_slice(&shape_i64).to_vec();

        graph.add_tensor(TensorInfo {
            name: shape_name.clone(),
            shape: TensorShape::Static(vec![out_shape.len()]),
            dtype: DataType::I64,
            kind: TensorKind::Intermediate,
            initializer: Some(shape_bytes),
        });

        graph.add_node(
            Node::new("Reshape")
                .with_inputs(vec![previous_output, shape_name])
                .with_outputs(vec![output.clone()]),
        );

        previous_output = output;
    }

    graph.outputs = vec![previous_output];
    graph
}

// ============================================================================
// Benchmark infrastructure
// ============================================================================

fn executor(graph: &Graph) -> DispatchExecutor {
    let registry = core_operator_registry();
    let runtime = block_on(Runtime::new()).expect("should get a runtime");
    let model = compile(graph, &registry, runtime.gpu()).expect("graph should compile");
    runtime.load_model(model).expect("should get executor")
}

struct BenchCtx {
    executor: DispatchExecutor,
    inputs: Vec<(&'static str, Tensor)>,
}

impl BenchCtx {
    fn from_graph(graph: &Graph, inputs: Vec<(&'static str, Tensor)>) -> Self {
        let executor = executor(graph);
        Self { executor, inputs }
    }

    fn execute(&mut self) {
        self.executor
            .run(&self.inputs)
            .expect("graph should execute");
    }
}

// ============================================================================
// Benchmark functions
// ============================================================================

fn bench_unary_ops(c: &mut Criterion) {
    let unary_ops = ["Neg", "Sqrt", "Cos", "Sin", "Tanh", "Gelu"];

    for op in unary_ops {
        let mut group = c.benchmark_group(format!("operators/{op}"));
        group.sample_size(10);

        for io_size in [256, 4096, 65_536] {
            let graph = unary_op_graph(op, 100, io_size);
            group.bench_with_input(
                BenchmarkId::from_parameter(io_size),
                &io_size,
                |b, &io_size| {
                    b.iter_batched(
                        || {
                            BenchCtx::from_graph(
                                &graph,
                                vec![("input", Tensor::new_1d_splat(1., io_size))],
                            )
                        },
                        |mut ctx| ctx.execute(),
                        criterion::BatchSize::LargeInput,
                    );
                },
            );
        }
    }
}

fn bench_binary_ops(c: &mut Criterion) {
    let binary_ops = ["Add", "Mul", "Div", "Sub", "Pow"];

    for op in binary_ops {
        let mut group = c.benchmark_group(format!("operators/{op}"));
        group.sample_size(10);

        for io_size in [256, 4096, 65_536] {
            let graph = binary_op_graph(op, 100, io_size);
            group.bench_with_input(
                BenchmarkId::from_parameter(io_size),
                &io_size,
                |b, &io_size| {
                    b.iter_batched(
                        || {
                            BenchCtx::from_graph(
                                &graph,
                                vec![
                                    ("input1", Tensor::new_1d_splat(2., io_size)),
                                    ("input2", Tensor::new_1d_splat(0.5, io_size)),
                                ],
                            )
                        },
                        |mut ctx| ctx.execute(),
                        criterion::BatchSize::LargeInput,
                    );
                },
            );
        }
    }
}

fn bench_comparison_ops(c: &mut Criterion) {
    let comparison_ops = ["Equal", "Greater", "Less", "GreaterOrEqual", "LessOrEqual"];

    for op in comparison_ops {
        let mut group = c.benchmark_group(format!("operators/{op}"));
        group.sample_size(10);

        for io_size in [256, 4096, 65_536] {
            let graph = binary_op_graph(op, 100, io_size);
            group.bench_with_input(
                BenchmarkId::from_parameter(io_size),
                &io_size,
                |b, &io_size| {
                    b.iter_batched(
                        || {
                            BenchCtx::from_graph(
                                &graph,
                                vec![
                                    ("input1", Tensor::new_1d_splat(1.5, io_size)),
                                    ("input2", Tensor::new_1d_splat(1.0, io_size)),
                                ],
                            )
                        },
                        |mut ctx| ctx.execute(),
                        criterion::BatchSize::LargeInput,
                    );
                },
            );
        }
    }
}

fn bench_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("operators/MatMul");
    group.sample_size(10);

    // Test different matrix sizes: (M, K) × (K, N)
    let sizes = [
        (64, 64, 64),
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
    ];

    for (m, k, n) in sizes {
        let graph = matmul_graph(10, m, k, n);
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{m}×{k}×{n}")),
            &(m, k, n),
            |b, &(m, k, n)| {
                b.iter_batched(
                    || {
                        BenchCtx::from_graph(
                            &graph,
                            vec![
                                ("input_a", Tensor::new_2d_splat(1., m, k)),
                                ("input_b", Tensor::new_2d_splat(1., k, n)),
                            ],
                        )
                    },
                    |mut ctx| ctx.execute(),
                    criterion::BatchSize::LargeInput,
                );
            },
        );
    }
}

fn bench_reduction_ops(c: &mut Criterion) {
    let reduction_ops = ["ReduceMean", "ReduceSum"];

    for op in reduction_ops {
        let mut group = c.benchmark_group(format!("operators/{op}"));
        group.sample_size(10);

        // Test reduction over different dimensions
        let configs = [
            (vec![256, 256], vec![1i64]),       // Reduce over last dim
            (vec![64, 64, 64], vec![2i64]),     // Reduce over last dim
            (vec![32, 32, 32, 32], vec![3i64]), // Reduce over last dim
        ];

        for (shape, axes) in &configs {
            let graph = reduction_graph(op, 50, shape.clone(), axes.clone());

            group.bench_with_input(
                BenchmarkId::from_parameter(format!("{:?}", shape)),
                shape,
                |b, shape| {
                    b.iter_batched(
                        || {
                            BenchCtx::from_graph(
                                &graph,
                                vec![("input", Tensor::new_nd_splat(1., shape.clone()))],
                            )
                        },
                        |mut ctx| ctx.execute(),
                        criterion::BatchSize::LargeInput,
                    );
                },
            );
        }
    }
}

fn bench_softmax(c: &mut Criterion) {
    let mut group = c.benchmark_group("operators/Softmax");
    group.sample_size(10);

    // Test softmax over different shapes and axes
    let configs = [
        (vec![128, 1024], -1i64),  // Batch softmax
        (vec![32, 32, 32], -1i64), // 3D softmax
        (vec![16, 64, 64], 1i64),  // Middle dimension
    ];

    for (shape, axis) in &configs {
        let graph = softmax_graph(50, shape.clone(), *axis);

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{:?}", shape)),
            shape,
            |b, shape| {
                b.iter_batched(
                    || {
                        BenchCtx::from_graph(
                            &graph,
                            vec![("input", Tensor::new_nd_splat(1., shape.clone()))],
                        )
                    },
                    |mut ctx| ctx.execute(),
                    criterion::BatchSize::LargeInput,
                );
            },
        );
    }
}

fn bench_transpose(c: &mut Criterion) {
    let mut group = c.benchmark_group("operators/Transpose");
    group.sample_size(10);

    // Test transpose over different shapes
    let configs = [
        (vec![256, 256], vec![1i64, 0]),             // 2D transpose
        (vec![64, 64, 64], vec![2i64, 1, 0]),        // 3D reverse
        (vec![32, 32, 32, 32], vec![0i64, 2, 1, 3]), // Swap middle dims
    ];

    for (shape, perm) in &configs {
        let graph = transpose_graph(50, shape.clone(), perm.clone());

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{:?}", shape)),
            shape,
            |b, shape| {
                b.iter_batched(
                    || {
                        BenchCtx::from_graph(
                            &graph,
                            vec![("input", Tensor::new_nd_splat(1., shape.clone()))],
                        )
                    },
                    |mut ctx| ctx.execute(),
                    criterion::BatchSize::LargeInput,
                );
            },
        );
    }
}

fn bench_cast(c: &mut Criterion) {
    let mut group = c.benchmark_group("operators/Cast");
    group.sample_size(10);

    for io_size in [256, 4096, 65_536] {
        let graph = cast_graph(50, io_size);
        group.bench_with_input(
            BenchmarkId::from_parameter(io_size),
            &io_size,
            |b, &io_size| {
                b.iter_batched(
                    || {
                        BenchCtx::from_graph(
                            &graph,
                            vec![("input", Tensor::new_1d_splat(1.5, io_size))],
                        )
                    },
                    |mut ctx| ctx.execute(),
                    criterion::BatchSize::LargeInput,
                );
            },
        );
    }
}

fn bench_where(c: &mut Criterion) {
    let mut group = c.benchmark_group("operators/Where");
    group.sample_size(10);

    for io_size in [256, 4096, 65_536] {
        let graph = where_graph(50, io_size);
        group.bench_with_input(
            BenchmarkId::from_parameter(io_size),
            &io_size,
            |b, &io_size| {
                b.iter_batched(
                    || {
                        // Create a bool tensor: true for odd indices, false for even
                        let cond_data: Vec<bool> = (0..io_size).map(|i| i % 2 == 1).collect();
                        let cond_tensor = Tensor::new_bool_1d(cond_data, io_size);

                        BenchCtx::from_graph(
                            &graph,
                            vec![
                                ("cond", cond_tensor),
                                ("x", Tensor::new_1d_splat(1., io_size)),
                                ("y", Tensor::new_1d_splat(0., io_size)),
                            ],
                        )
                    },
                    |mut ctx| ctx.execute(),
                    criterion::BatchSize::LargeInput,
                );
            },
        );
    }
}

fn bench_reshape(c: &mut Criterion) {
    let mut group = c.benchmark_group("operators/Reshape");
    group.sample_size(10);

    let configs = [
        vec![256, 256],       // 2D
        vec![64, 64, 64],     // 3D
        vec![32, 32, 32, 32], // 4D
    ];

    for shape in &configs {
        let graph = reshape_graph(50, shape.clone());

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{:?}", shape)),
            shape,
            |b, shape| {
                b.iter_batched(
                    || {
                        BenchCtx::from_graph(
                            &graph,
                            vec![("input", Tensor::new_nd_splat(1., shape.clone()))],
                        )
                    },
                    |mut ctx| ctx.execute(),
                    criterion::BatchSize::LargeInput,
                );
            },
        );
    }
}

criterion_group!(
    benches,
    bench_unary_ops,
    bench_binary_ops,
    bench_comparison_ops,
    bench_matmul,
    bench_reduction_ops,
    bench_softmax,
    bench_transpose,
    bench_cast,
    bench_where,
    bench_reshape
);
criterion_main!(benches);
