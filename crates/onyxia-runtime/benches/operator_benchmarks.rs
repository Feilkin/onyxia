use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use pollster::block_on;

use onyxia_compiler::compile;
use onyxia_onnx::{DataType, Graph, Node, TensorInfo, TensorKind, TensorShape};
use onyxia_operators::core_operator_registry;
use onyxia_runtime::{DispatchExecutor, Runtime, Tensor};

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

fn executor(graph: &Graph) -> DispatchExecutor {
    let registry = core_operator_registry();
    let model = compile(graph, &registry).expect("graph should compile");
    let runtime = block_on(Runtime::new()).expect("should get a runtime");
    block_on(runtime.load_model(model)).expect("should get executor")
}

fn bench_unary_ops(c: &mut Criterion) {
    let unary_ops = ["Neg", "Sqrt", "Cos", "Sin", "Tanh", "Gelu"];

    for op in unary_ops {
        let mut group = c.benchmark_group(format!("operators {op}"));
        group.sample_size(10);

        for io_size in [256, 1024, 4096, 65_536] {
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

criterion_group!(benches, bench_unary_ops);
criterion_main!(benches);
