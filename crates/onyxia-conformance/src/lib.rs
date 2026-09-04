//! ONNX operator conformance: run the official `onnx` node tests
//! (`onnx/backend/test/data/node/*`, Apache-2.0) through lowering and a
//! backend, and compare against the expected outputs.
//!
//! Each node test is a directory holding `model.onnx` plus one or more
//! `test_data_set_N/` directories of `input_K.pb` / `output_K.pb`
//! `TensorProto` files. The harness discovers them, classifies each result
//! as pass / fail / skip (with a reason), and aggregates by operator so
//! the coverage matrix is one command away.
//!
//! Test data location, in order: `$ONNX_NODE_TESTS`, then the
//! `onnx` package inside a `.venv` at the workspace root (`just
//! fetch-onnx-tests` installs it).

use onyxia_ir::interp::Tensor;
use onyxia_ir::{DataType, Module};
use onyxia_onnx::{AttrTensor, TensorProto, parse_tensor_proto};
use prost::Message;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// The `ai.onnx` operator list from the vendored spec (opset 26).
pub const OPS: &str = include_str!("ops.txt");

/// One discovered node test.
#[derive(Debug, Clone)]
pub struct NodeTest {
    /// Directory name, e.g. `test_add_bcast`.
    pub name: String,
    pub dir: PathBuf,
}

/// A loaded data set: named inputs and expected outputs.
pub struct DataSet {
    pub inputs: Vec<(String, Tensor)>,
    pub outputs: Vec<(String, Tensor)>,
}

/// Outcome of one node test.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    Pass,
    /// Passed, but only after the small parameter inputs (axes, shapes,
    /// slice bounds, …) were bound as constants — the node tests feed
    /// these as runtime inputs, real exports carry them as initializers.
    PassBound,
    /// Something we claim to support produced the wrong answer (or
    /// crashed): lowering, preparation, execution, or a mismatch.
    Fail(String),
    /// Out of scope for the harness: an unsupported dtype, a non-tensor
    /// type, or an op with no lowering rule.
    Skip(String),
}

impl Outcome {
    pub fn is_pass(&self) -> bool {
        matches!(self, Outcome::Pass | Outcome::PassBound)
    }
}

/// A backend the harness can run a lowered module on.
pub trait Runner {
    fn name(&self) -> &str;
    fn run(
        &mut self,
        module: Module,
        inputs: &[(&str, Tensor)],
    ) -> onyxia_ir::Result<Vec<(String, Tensor)>>;
}

/// The reference interpreter.
pub struct RefRunner;

impl Runner for RefRunner {
    fn name(&self) -> &str {
        "ref"
    }
    fn run(
        &mut self,
        module: Module,
        inputs: &[(&str, Tensor)],
    ) -> onyxia_ir::Result<Vec<(String, Tensor)>> {
        onyxia_backend_ref::run_once(module, inputs)
    }
}

#[cfg(feature = "wgpu")]
pub struct WgpuRunner {
    ctx: onyxia_backend_wgpu::GpuContext,
}

#[cfg(feature = "wgpu")]
impl WgpuRunner {
    pub fn new() -> onyxia_ir::Result<Self> {
        let ctx = pollster::block_on(onyxia_backend_wgpu::GpuContext::new())?;
        Ok(Self { ctx })
    }
}

#[cfg(feature = "wgpu")]
impl Runner for WgpuRunner {
    fn name(&self) -> &str {
        "wgpu"
    }
    fn run(
        &mut self,
        module: Module,
        inputs: &[(&str, Tensor)],
    ) -> onyxia_ir::Result<Vec<(String, Tensor)>> {
        use onyxia_ir::{Backend, Session};
        let backend = onyxia_backend_wgpu::WgpuBackend::new(self.ctx.clone());
        let mut session = backend.prepare(module)?;
        let dev: Vec<(&str, _)> = inputs
            .iter()
            .map(|(n, t)| Ok((*n, session.upload(t)?)))
            .collect::<onyxia_ir::Result<_>>()?;
        pollster::block_on(async {
            let outs = session.run(&dev).await?;
            let mut host = Vec::new();
            for (n, t) in outs {
                host.push((n, session.download(&t).await?));
            }
            Ok(host)
        })
    }
}

// ───────────────────────────── discovery ────────────────────────────────

/// Locate the node test directory.
pub fn find_data_dir() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("ONNX_NODE_TESTS") {
        let p = PathBuf::from(p);
        if p.is_dir() {
            return Some(p);
        }
    }
    let root = workspace_root();
    let venv = root.join(".venv/lib");
    if let Ok(entries) = std::fs::read_dir(&venv) {
        for e in entries.flatten() {
            let p = e.path().join("site-packages/onnx/backend/test/data/node");
            if p.is_dir() {
                return Some(p);
            }
        }
    }
    None
}

fn workspace_root() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.to_path_buf())
        .unwrap_or(manifest)
}

/// Discover node tests, sorted by name.
pub fn discover(dir: &Path) -> std::io::Result<Vec<NodeTest>> {
    let mut tests = Vec::new();
    for e in std::fs::read_dir(dir)? {
        let e = e?;
        let path = e.path();
        if path.is_dir() && path.join("model.onnx").is_file() {
            tests.push(NodeTest {
                name: e.file_name().to_string_lossy().into_owned(),
                dir: path,
            });
        }
    }
    tests.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(tests)
}

// ────────────────────────────── loading ─────────────────────────────────

fn read_tensor_pb(path: &Path) -> Result<AttrTensor, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let proto =
        TensorProto::decode(bytes.as_slice()).map_err(|e| format!("{}: {e}", path.display()))?;
    parse_tensor_proto(&proto).map_err(|e| e.to_string())
}

fn to_tensor(t: &AttrTensor) -> Result<Tensor, String> {
    let dtype = convert_dtype(t.dtype);
    Tensor::new(dtype, t.dims.clone(), t.data.clone()).map_err(|e| e.to_string())
}

fn convert_dtype(dt: onyxia_onnx::DataType) -> DataType {
    use onyxia_onnx::DataType as D;
    match dt {
        D::F32 => DataType::F32,
        D::F16 => DataType::F16,
        D::I32 => DataType::I32,
        D::I64 => DataType::I64,
        D::U8 => DataType::U8,
        D::I8 => DataType::I8,
        D::U32 => DataType::U32,
        D::Bool => DataType::Bool,
        D::Q4 => DataType::U4,
        D::Q8 => DataType::U8,
    }
}

/// Load every `test_data_set_*` of a node test. Tensor names come from
/// the model's declared inputs/outputs, positionally.
pub fn load_data_sets(test: &NodeTest, graph: &onyxia_onnx::Graph) -> Result<Vec<DataSet>, String> {
    let mut sets = Vec::new();
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(&test.dir)
        .map_err(|e| e.to_string())?
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.is_dir()
                && p.file_name()
                    .map(|n| n.to_string_lossy().starts_with("test_data_set_"))
                    .unwrap_or(false)
        })
        .collect();
    dirs.sort();
    for d in dirs {
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        for (i, name) in graph.inputs.iter().enumerate() {
            let p = d.join(format!("input_{i}.pb"));
            if !p.is_file() {
                break;
            }
            inputs.push((name.clone(), to_tensor(&read_tensor_pb(&p)?)?));
        }
        for (i, name) in graph.outputs.iter().enumerate() {
            let p = d.join(format!("output_{i}.pb"));
            if !p.is_file() {
                return Err(format!("missing {}", p.display()));
            }
            outputs.push((name.clone(), to_tensor(&read_tensor_pb(&p)?)?));
        }
        sets.push(DataSet { inputs, outputs });
    }
    if sets.is_empty() {
        return Err("no test_data_set_* directories".into());
    }
    Ok(sets)
}

// ────────────────────────────── comparison ──────────────────────────────

/// Tolerances follow the onnx backend test defaults (rtol 1e-3) with a
/// slightly looser atol for f32 (their 1e-7 assumes numpy's f32 paths).
pub fn compare(expected: &Tensor, got: &Tensor) -> Result<(), String> {
    if expected.shape() != got.shape() {
        return Err(format!(
            "shape {:?} expected, got {:?}",
            expected.shape(),
            got.shape()
        ));
    }
    if expected.dtype() != got.dtype() {
        return Err(format!(
            "dtype {} expected, got {}",
            expected.dtype(),
            got.dtype()
        ));
    }
    match expected.dtype() {
        DataType::F32 | DataType::F16 => {
            let (rtol, atol) = if expected.dtype() == DataType::F16 {
                (1e-2f32, 1e-3f32)
            } else {
                (1e-3f32, 1e-5f32)
            };
            let e = expected.to_f32().map_err(|e| e.to_string())?;
            let g = got.to_f32().map_err(|e| e.to_string())?;
            let mut worst: Option<(usize, f32, f32)> = None;
            let mut bad = 0usize;
            for (i, (a, b)) in e.iter().zip(&g).enumerate() {
                let ok = if a.is_nan() {
                    b.is_nan()
                } else if a.is_infinite() {
                    a == b
                } else {
                    (a - b).abs() <= atol + rtol * a.abs()
                };
                if !ok {
                    bad += 1;
                    let err = (a - b).abs();
                    if worst.map_or(true, |(_, wa, wb)| err > (wa - wb).abs()) {
                        worst = Some((i, *a, *b));
                    }
                }
            }
            if let Some((i, a, b)) = worst {
                return Err(format!(
                    "{bad}/{} elements differ; worst at [{i}]: expected {a}, got {b}",
                    e.len()
                ));
            }
            Ok(())
        }
        DataType::Bool => {
            let e = expected.to_bool().map_err(|e| e.to_string())?;
            let g = got.to_bool().map_err(|e| e.to_string())?;
            first_mismatch(&e, &g)
        }
        _ => {
            let e = expected.to_i64().map_err(|e| e.to_string())?;
            let g = got.to_i64().map_err(|e| e.to_string())?;
            first_mismatch(&e, &g)
        }
    }
}

fn first_mismatch<T: PartialEq + std::fmt::Debug>(e: &[T], g: &[T]) -> Result<(), String> {
    let bad = e.iter().zip(g).filter(|(a, b)| a != b).count();
    if let Some((i, (a, b))) = e.iter().zip(g).enumerate().find(|(_, (a, b))| a != b) {
        return Err(format!(
            "{bad}/{} elements differ; first at [{i}]: expected {a:?}, got {b:?}",
            e.len()
        ));
    }
    Ok(())
}

// ─────────────────────────────── running ────────────────────────────────

/// Classify an error message as skip (out of scope) or fail.
fn classify(stage: &str, msg: String) -> Outcome {
    let m = msg.to_ascii_lowercase();
    let skip = m.contains("unsupported data type")
        || m.contains("non-tensor type")
        || m.contains("no lowering rule")
        || m.contains("onnx tensor data type")
        || m.contains("double/string typed data")
        || m.contains("int32_data with data type");
    if skip {
        Outcome::Skip(format!("{stage}: {msg}"))
    } else {
        Outcome::Fail(format!("{stage}: {msg}"))
    }
}

/// Run one node test end to end.
pub fn run_test(test: &NodeTest, runner: &mut dyn Runner) -> Outcome {
    let model = match onyxia_onnx::load_model(test.dir.join("model.onnx")) {
        Ok(m) => m,
        Err(e) => return classify("load", e.to_string()),
    };
    let mut graph = match onyxia_onnx::parse_model(&model, Some(&test.dir)) {
        Ok(g) => g,
        Err(e) => return classify("parse", e.to_string()),
    };
    let sets = match load_data_sets(test, &graph) {
        Ok(s) => s,
        Err(e) => return classify("data", e),
    };
    let registry = onyxia_lower::standard_registry();
    let mut bound = false;
    let module = match onyxia_lower::lower(graph.clone(), &registry) {
        Ok(m) => m,
        Err(first) => {
            // Parameter inputs fed as runtime tensors: retry with every
            // small (≤ 64 elements, rank ≤ 1) input of the first data set
            // bound as an initializer.
            let retry = sets.len() == 1 && bind_small_inputs(&mut graph, &sets[0]);
            let lowered = if retry {
                onyxia_lower::lower(graph, &registry)
            } else {
                Err(first)
            };
            match lowered {
                Ok(m) => {
                    bound = true;
                    m
                }
                Err(e) => return classify("lower", e.to_string()),
            }
        }
    };
    for (si, set) in sets.iter().enumerate() {
        let inputs: Vec<(&str, Tensor)> = set
            .inputs
            .iter()
            .filter(|(n, _)| module.inputs.iter().any(|(m, _)| m == n))
            .map(|(n, t)| (n.as_str(), t.clone()))
            .collect();
        let got = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            runner.run(module.clone(), &inputs)
        })) {
            Ok(Ok(g)) => g,
            Ok(Err(e)) => return classify("run", e.to_string()),
            Err(p) => {
                let msg = p
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_else(|| "panic".into());
                return Outcome::Fail(format!("run: panic: {msg}"));
            }
        };
        for (name, expected) in &set.outputs {
            let Some((_, g)) = got.iter().find(|(n, _)| n == name) else {
                return Outcome::Fail(format!("output '{name}' missing"));
            };
            if let Err(e) = compare(expected, g) {
                return Outcome::Fail(format!("set {si} output '{name}': {e}"));
            }
        }
    }
    if bound {
        Outcome::PassBound
    } else {
        Outcome::Pass
    }
}

/// Turn the small inputs of `set` into initializers of `graph`. Returns
/// whether anything changed.
fn bind_small_inputs(graph: &mut onyxia_onnx::Graph, set: &DataSet) -> bool {
    let mut changed = false;
    for (name, t) in &set.inputs {
        if t.shape().len() > 1 || t.numel() > 64 {
            continue;
        }
        let Ok(id) = graph.tensor_id(name) else {
            continue;
        };
        let info = &mut graph.tensor_info[id];
        info.kind = onyxia_onnx::TensorKind::Weight;
        info.shape = onyxia_onnx::TensorShape::Static(t.shape().to_vec());
        info.initializer = Some(t.bytes().to_vec());
        graph.inputs.retain(|n| n != name);
        changed = true;
    }
    changed
}

/// Test names the harness never attempts: op families outside the tensor
/// model (sequences, optionals, strings, control flow, training, random).
pub fn out_of_scope(name: &str) -> Option<&'static str> {
    let n = name.strip_prefix("test_").unwrap_or(name);
    const GROUPS: &[(&str, &[&str])] = &[
        (
            "sequence/optional types",
            &[
                "sequence_",
                "optional_",
                "split_to_sequence",
                "concat_from_sequence",
                "sequence",
            ],
        ),
        (
            "string ops",
            &["string", "strnormalizer", "regex_", "tfidfvectorizer"],
        ),
        (
            "control flow (If/Loop/Scan)",
            &["if_", "if", "loop", "scan_", "scan9", "scan"],
        ),
        (
            "training ops",
            &[
                "adam",
                "adagrad",
                "momentum",
                "nesterov",
                "gradient",
                "training_",
            ],
        ),
        ("random ops", &["bernoulli", "random", "multinomial"]),
        ("image decoding", &["image_decoder"]),
        ("ai.onnx.ml domain", &["ai_onnx_ml"]),
    ];
    for (why, prefixes) in GROUPS {
        if prefixes.iter().any(|p| n.starts_with(p)) {
            return Some(why);
        }
    }
    None
}

// ─────────────────────────────── reporting ──────────────────────────────

/// Guess the operator under test from the test name: longest `ai.onnx`
/// op name (case-insensitive, underscores ignored) that prefixes it.
pub fn op_of(test_name: &str) -> String {
    let n: String = test_name
        .strip_prefix("test_")
        .unwrap_or(test_name)
        .chars()
        .filter(|c| *c != '_')
        .collect::<String>()
        .to_ascii_lowercase();
    const ALIASES: &[(&str, &str)] = &[
        ("sce", "SoftmaxCrossEntropyLoss"),
        ("nllloss", "NegativeLogLikelihoodLoss"),
        ("mvn", "MeanVarianceNormalization"),
        ("batchnorm", "BatchNormalization"),
        ("instancenorm", "InstanceNormalization"),
        ("groupnorm", "GroupNormalization"),
        ("triu", "Trilu"),
        ("tril", "Trilu"),
        ("basicconv", "Conv"),
        ("simpleconv", "Conv"),
        ("basicdeformconv", "DeformConv"),
        ("deformconv", "DeformConv"),
        ("convinteger", "ConvInteger"),
        ("convtranspose", "ConvTranspose"),
        ("qlinearconv", "QLinearConv"),
        ("qlinearmatmul", "QLinearMatMul"),
        ("matmulinteger", "MatMulInteger"),
        ("logsoftmax", "LogSoftmax"),
        ("gatherelements", "GatherElements"),
        ("gathernd", "GatherND"),
        ("scatterelements", "ScatterElements"),
        ("scatternd", "ScatterND"),
        ("tensorscatter", "TensorScatter"),
        ("reflectpad", "Pad"),
        ("edgepad", "Pad"),
        ("wrappad", "Pad"),
        ("constantpad", "Pad"),
        ("centercroppad", "CenterCropPad"),
        ("top", "TopK"),
        ("upsample", "Upsample"),
        ("l1normalization", "LpNormalization"),
        ("l2normalization", "LpNormalization"),
        ("lpnormalization", "LpNormalization"),
        ("flexattention", "Attention"),
        ("causal", "Attention"),
        ("ai.onnx.ml", "ai.onnx.ml"),
        ("dynamicquantizelinear", "DynamicQuantizeLinear"),
        ("dequantizelinear", "DequantizeLinear"),
        ("quantizelinear", "QuantizeLinear"),
        ("globalaveragepool", "GlobalAveragePool"),
        ("globalmaxpool", "GlobalMaxPool"),
        ("hannwindow", "HannWindow"),
        ("hammingwindow", "HammingWindow"),
        ("blackmanwindow", "BlackmanWindow"),
        ("melweightmatrix", "MelWeightMatrix"),
        ("affinegrid", "AffineGrid"),
        ("gridsample", "GridSample"),
        ("roialign", "RoiAlign"),
        ("bitshift", "BitShift"),
        ("bitwise", "Bitwise"),
        ("bitcast", "BitCast"),
        ("hardsigmoid", "HardSigmoid"),
        ("hardswish", "HardSwish"),
        ("hardmax", "Hardmax"),
        ("shrink", "Shrink"),
        ("isinf", "IsInf"),
        ("isnan", "IsNaN"),
        ("cumsum", "CumSum"),
        ("cumprod", "CumProd"),
        ("constantofshape", "ConstantOfShape"),
        ("elu", "Elu"),
        ("prelu", "PRelu"),
        ("selu", "Selu"),
        ("celu", "Celu"),
        ("leakyrelu", "LeakyRelu"),
        ("thresholdedrelu", "ThresholdedRelu"),
        ("relu", "Relu"),
        ("reversesequence", "ReverseSequence"),
        ("nonmaxsuppression", "NonMaxSuppression"),
        ("nonzero", "NonZero"),
        ("eyelike", "EyeLike"),
        ("onehot", "OneHot"),
        ("rotaryembedding", "RotaryEmbedding"),
        ("rmsnormalization", "RMSNormalization"),
        ("layernormalization", "LayerNormalization"),
        ("spacetodepth", "SpaceToDepth"),
        ("depthtospace", "DepthToSpace"),
        ("col2im", "Col2Im"),
        ("maxunpool", "MaxUnpool"),
        ("maxpool", "MaxPool"),
        ("averagepool", "AveragePool"),
        ("lppool", "LpPool"),
        ("dropout", "Dropout"),
        ("castlike", "CastLike"),
        ("stft", "STFT"),
        ("dft", "DFT"),
    ];
    let mut best: Option<(&str, usize)> = None;
    for (alias, op) in ALIASES {
        if n.starts_with(alias) && best.map_or(true, |(_, l)| alias.len() > l) {
            best = Some((op, alias.len()));
        }
    }
    for op in OPS.lines() {
        let key: String = op
            .chars()
            .filter(|c| *c != '_')
            .collect::<String>()
            .to_ascii_lowercase();
        if n.starts_with(&key) && best.map_or(true, |(_, l)| key.len() > l) {
            best = Some((op, key.len()));
        }
    }
    best.map(|(op, _)| op.to_string())
        .unwrap_or_else(|| "?".into())
}

/// Per-op tallies.
#[derive(Debug, Default, Clone)]
pub struct OpStats {
    pub pass: usize,
    /// Of `pass`, how many needed parameter inputs bound as constants.
    pub bound: usize,
    pub fail: usize,
    pub skip: usize,
}

/// Aggregate outcomes by op.
pub fn by_op<'a>(
    results: impl Iterator<Item = (&'a str, &'a Outcome)>,
) -> BTreeMap<String, OpStats> {
    let mut m: BTreeMap<String, OpStats> = BTreeMap::new();
    for (name, outcome) in results {
        let e = m.entry(op_of(name)).or_default();
        match outcome {
            Outcome::Pass => e.pass += 1,
            Outcome::PassBound => {
                e.pass += 1;
                e.bound += 1;
            }
            Outcome::Fail(_) => e.fail += 1,
            Outcome::Skip(_) => e.skip += 1,
        }
    }
    m
}

/// Path of the checked-in list of node tests expected to pass on a backend.
pub fn expected_path(backend: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("expected-pass-{backend}.txt"))
}

/// Read an expected-pass list (comments and blanks ignored).
pub fn read_expected(backend: &str) -> Vec<String> {
    std::fs::read_to_string(expected_path(backend))
        .unwrap_or_default()
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(String::from)
        .collect()
}
