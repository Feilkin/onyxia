//! Regression gate: every node test listed in `expected-pass-ref.txt` must
//! still pass on the reference backend. Newly passing tests are reported
//! (add them with `cargo run -p onyxia-conformance -- --quiet --update-expected`).
//!
//! Without the test data (see the crate docs) this test is a no-op.

use onyxia_conformance::{
    Outcome, RefRunner, discover, find_data_dir, out_of_scope, read_expected, run_test,
};

#[test]
fn expected_node_tests_pass_on_ref() {
    let Some(dir) = find_data_dir() else {
        eprintln!("onnx node test data not found; skipping");
        return;
    };
    let expected = read_expected("ref");
    assert!(!expected.is_empty(), "expected-pass-ref.txt is empty");
    let tests = discover(&dir).unwrap();
    let mut runner = RefRunner;
    let mut regressions = Vec::new();
    let mut newly_passing = Vec::new();
    for t in &tests {
        if out_of_scope(&t.name).is_some() {
            continue;
        }
        let listed = expected.iter().any(|e| e == &t.name);
        let outcome = run_test(t, &mut runner);
        match (listed, &outcome) {
            (true, Outcome::Pass) => {}
            (true, other) => regressions.push(format!("{}: {other:?}", t.name)),
            (false, Outcome::Pass) => newly_passing.push(t.name.clone()),
            (false, _) => {}
        }
    }
    if !newly_passing.is_empty() {
        eprintln!(
            "{} node tests newly pass (add them with --update-expected):\n  {}",
            newly_passing.len(),
            newly_passing.join("\n  ")
        );
    }
    assert!(
        regressions.is_empty(),
        "regressions:\n  {}",
        regressions.join("\n  ")
    );
}
