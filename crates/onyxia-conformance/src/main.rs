//! `onyxia-conformance`: run the onnx node tests and print the coverage
//! matrix.
//!
//! ```text
//! onyxia-conformance [--backend ref|wgpu] [--filter SUBSTR]... [--ops]
//!                    [--failures] [--skips] [--update-expected]
//! ```

use onyxia_conformance::{
    Outcome, RefRunner, Runner, by_op, discover, find_data_dir, out_of_scope, run_test,
};
use std::collections::BTreeMap;

fn main() {
    let mut args = std::env::args().skip(1);
    let mut backend = "ref".to_string();
    let mut filters: Vec<String> = Vec::new();
    let (mut show_ops, mut show_failures, mut show_skips, mut update, mut quiet, mut dump) =
        (false, false, false, false, false, false);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--backend" => backend = args.next().expect("--backend NAME"),
            "--filter" => filters.push(args.next().expect("--filter SUBSTR")),
            "--ops" => show_ops = true,
            "--failures" => show_failures = true,
            "--skips" => show_skips = true,
            "--update-expected" => update = true,
            "--quiet" => quiet = true,
            "--dump-graph" => dump = true,
            other => filters.push(other.to_string()),
        }
    }

    let Some(dir) = find_data_dir() else {
        eprintln!(
            "no onnx node test data found: set ONNX_NODE_TESTS or run `just fetch-onnx-tests`"
        );
        std::process::exit(2);
    };
    let tests = discover(&dir).expect("discover");
    let tests: Vec<_> = tests
        .into_iter()
        .filter(|t| filters.is_empty() || filters.iter().any(|f| t.name.contains(f.as_str())))
        .collect();

    let mut runner: Box<dyn Runner> = match backend.as_str() {
        "ref" => Box::new(RefRunner),
        #[cfg(feature = "wgpu")]
        "wgpu" => Box::new(onyxia_conformance::WgpuRunner::new().expect("GPU")),
        other => {
            eprintln!("unknown backend '{other}' (build with --features wgpu for the GPU backend)");
            std::process::exit(2);
        }
    };

    // Silence panics: the harness reports them as failures. Set
    // RUST_BACKTRACE to keep the default hook for debugging.
    if std::env::var_os("RUST_BACKTRACE").is_none() {
        std::panic::set_hook(Box::new(|_| {}));
    }

    let mut results: Vec<(String, Outcome)> = Vec::with_capacity(tests.len());
    for t in &tests {
        if dump {
            onyxia_conformance::dump_graph(t);
        }
        let outcome = match out_of_scope(&t.name) {
            Some(why) => Outcome::Skip(format!("out of scope: {why}")),
            None => run_test(t, runner.as_mut()),
        };
        if !quiet {
            let tag = match &outcome {
                Outcome::Pass => "PASS",
                Outcome::PassBound => "PASS*",
                Outcome::Fail(_) => "FAIL",
                Outcome::Skip(_) => "skip",
            };
            match &outcome {
                Outcome::Fail(m) if show_failures => println!("{tag} {}\n      {m}", t.name),
                Outcome::Skip(m) if show_skips => println!("{tag} {}\n      {m}", t.name),
                _ => println!("{tag} {}", t.name),
            }
        }
        results.push((t.name.clone(), outcome));
    }

    let pass = results.iter().filter(|(_, o)| o.is_pass()).count();
    let bound = results
        .iter()
        .filter(|(_, o)| matches!(o, Outcome::PassBound))
        .count();
    let fail = results
        .iter()
        .filter(|(_, o)| matches!(o, Outcome::Fail(_)))
        .count();
    let skip = results.len() - pass - fail;
    println!(
        "\n{} backend: {pass} pass ({bound} with parameter inputs bound as constants), {fail} fail, {skip} skip of {} node tests",
        runner.name(),
        results.len()
    );

    if show_ops {
        let ops = by_op(results.iter().map(|(n, o)| (n.as_str(), o)));
        println!(
            "\n{:<28} {:>5} {:>5} {:>5}   {}",
            "op", "pass", "fail", "skip", "(bound)"
        );
        for (op, s) in &ops {
            let b = if s.bound > 0 {
                format!("  ({} bound)", s.bound)
            } else {
                String::new()
            };
            println!("{op:<28} {:>5} {:>5} {:>5}{b}", s.pass, s.fail, s.skip);
        }
        let full: Vec<&String> = ops
            .iter()
            .filter(|(_, s)| s.fail == 0 && s.pass > 0)
            .map(|(o, _)| o)
            .collect();
        let partial: Vec<&String> = ops
            .iter()
            .filter(|(_, s)| s.fail > 0 && s.pass > 0)
            .map(|(o, _)| o)
            .collect();
        let none: Vec<&String> = ops
            .iter()
            .filter(|(_, s)| s.pass == 0)
            .map(|(o, _)| o)
            .collect();
        println!(
            "\n{} ops fully passing, {} partially, {} not at all",
            full.len(),
            partial.len(),
            none.len()
        );
        // Skip reasons, aggregated.
        let mut reasons: BTreeMap<String, usize> = BTreeMap::new();
        for (_, o) in &results {
            if let Outcome::Skip(m) = o {
                let key = m.split(" (node").next().unwrap_or(m).to_string();
                let key = key
                    .split(": initializer")
                    .next()
                    .unwrap_or(&key)
                    .to_string();
                *reasons.entry(key).or_default() += 1;
            }
        }
        println!("\nskip reasons:");
        let mut v: Vec<_> = reasons.into_iter().collect();
        v.sort_by(|a, b| b.1.cmp(&a.1));
        for (r, n) in v.iter().take(40) {
            println!("{n:>5}  {r}");
        }
    }

    if update {
        let path = onyxia_conformance::expected_path(runner.name());
        let mut names: Vec<&str> = results
            .iter()
            .filter(|(_, o)| o.is_pass())
            .map(|(n, _)| n.as_str())
            .collect();
        names.sort_unstable();
        let body = format!(
            "# Node tests expected to pass on the {} backend. Regenerate with\n# `cargo run -p onyxia-conformance -- --quiet --update-expected`.\n{}\n",
            runner.name(),
            names.join("\n")
        );
        std::fs::write(&path, body).expect("write expected list");
        println!("wrote {} ({} tests)", path.display(), names.len());
    }
}
