//! Memory planning: liveness analysis for register-based execution.
//!
//! This module provides compile-time analysis of which registers become dead
//! after each dispatch step. The runtime uses this information to release GPU
//! buffers as soon as they are no longer needed, enabling buffer reuse via a
//! pool.

use crate::dispatch::CompiledModel;
use std::collections::HashMap;

/// Liveness information for the dispatch sequence.
///
/// For each dispatch entry, records which registers become dead after that
/// entry completes. A register is dead when it will never be read again.
pub struct LivenessInfo {
    /// For each entry index, which registers become dead after this entry.
    ///
    /// `freed_after[i]` contains the register indices whose last use is
    /// entry `i`. The runtime should release the tensors stored in these
    /// registers after entry `i` has executed.
    pub freed_after: Vec<Vec<usize>>,
}

/// Analyze register liveness across the dispatch sequence.
///
/// Walks the dispatch entries to determine the last use of each register.
/// A register's "last use" is the highest-indexed entry that reads it as
/// an input. Registers listed as model outputs are never freed.
///
/// # Algorithm
///
/// 1. Build `last_use: HashMap<usize, usize>` mapping each register to the
///    last entry index that reads it as input.
/// 2. Override all model-output registers to `usize::MAX` (never freed).
/// 3. Invert the map into `freed_after[entry_idx]`.
pub fn analyze_liveness(model: &CompiledModel) -> LivenessInfo {
    let n = model.entries.len();

    // Registers that must never be freed:
    //  - Model outputs: consumed by the caller after dispatch.
    //  - Weight registers: uploaded once at load time, never re-uploaded.
    //    Freeing them would corrupt subsequent inference runs.
    let mut pinned_regs: std::collections::HashSet<usize> =
        model.output_registers.iter().map(|(_, r)| *r).collect();
    for w in &model.weight_registers {
        pinned_regs.insert(w.register);
    }

    // Walk entries in forward order; update last_use[reg] for every input register.
    let mut last_use: HashMap<usize, usize> = HashMap::new();
    for (i, entry) in model.entries.iter().enumerate() {
        for &reg in &entry.input_regs {
            last_use.insert(reg, i);
        }
    }

    // Pinned registers are never freed.
    for &reg in &pinned_regs {
        last_use.insert(reg, usize::MAX);
    }

    // Invert last_use into freed_after.
    let mut freed_after = vec![Vec::new(); n];
    for (&reg, &last) in &last_use {
        if last != usize::MAX && last < n {
            freed_after[last].push(reg);
        }
    }

    LivenessInfo { freed_after }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dispatch::{CompiledModel, DispatchCtx, DispatchEntry, OpDispatch, RuntimeTensor};
    use crate::plan::ModelMetadata;
    use crate::types::DataType;

    /// Build a minimal CompiledModel from a list of (input_regs, output_regs) pairs.
    ///
    /// The model has `num_regs` registers total, with `output_regs` as model outputs.
    fn make_model(
        entries: &[(&[usize], &[usize])],
        num_regs: usize,
        model_outputs: &[usize],
    ) -> CompiledModel {
        struct DummyOp;
        impl OpDispatch for DummyOp {
            fn dispatch(
                &self,
                _inputs: Vec<RuntimeTensor>,
                _ctx: &mut DispatchCtx,
            ) -> crate::Result<Vec<RuntimeTensor>> {
                Ok(vec![])
            }
        }

        let dispatch_entries: Vec<DispatchEntry> = entries
            .iter()
            .map(|(ins, outs)| DispatchEntry {
                op: Box::new(DummyOp),
                input_regs: ins.to_vec(),
                output_regs: outs.to_vec(),
                name: "dummy".to_string(),
                node_name: String::new(),
            })
            .collect();

        let output_registers = model_outputs
            .iter()
            .map(|&r| (format!("out_{r}"), r))
            .collect();

        CompiledModel {
            entries: dispatch_entries,
            num_registers: num_regs,
            input_registers: vec![],
            output_registers,
            weight_registers: vec![],
            metadata: ModelMetadata::default(),
            liveness: None,
        }
    }

    /// Linear chain: reg0 → op0 → reg1 → op1 → reg2 → op2 → reg3 (output).
    ///
    /// Expected: reg0 freed after op0, reg1 freed after op1, reg2 freed after op2.
    /// reg3 is the model output and must never be freed.
    #[test]
    fn test_linear_chain() {
        // op0: reads reg0, writes reg1
        // op1: reads reg1, writes reg2
        // op2: reads reg2, writes reg3
        let model = make_model(&[(&[0], &[1]), (&[1], &[2]), (&[2], &[3])], 4, &[3]);
        let liveness = analyze_liveness(&model);
        assert_eq!(liveness.freed_after.len(), 3);

        // reg0: last read by op0 (index 0)
        assert!(liveness.freed_after[0].contains(&0), "reg0 freed after op0");
        // reg1: last read by op1 (index 1)
        assert!(liveness.freed_after[1].contains(&1), "reg1 freed after op1");
        // reg2: last read by op2 (index 2)
        assert!(liveness.freed_after[2].contains(&2), "reg2 freed after op2");

        // reg3 is a model output, must NOT appear in any freed_after
        let all_freed: Vec<usize> = liveness.freed_after.iter().flatten().copied().collect();
        assert!(!all_freed.contains(&3), "reg3 (output) must not be freed");
    }

    /// Fan-out: reg0 read by op0 and op1.
    ///
    /// Expected: reg0 freed after op1 (the later of the two uses).
    #[test]
    fn test_fan_out() {
        // op0: reads reg0, writes reg1
        // op1: reads reg0, writes reg2   ← reg0 last used here
        let model = make_model(&[(&[0], &[1]), (&[0], &[2])], 3, &[1, 2]);
        let liveness = analyze_liveness(&model);

        // reg0 last read by op1 (index 1)
        assert!(liveness.freed_after[1].contains(&0), "reg0 freed after op1");
        // reg0 must NOT be freed after op0
        assert!(
            !liveness.freed_after[0].contains(&0),
            "reg0 must not be freed after op0"
        );
    }

    /// Output registers are never freed.
    #[test]
    fn test_output_registers_never_freed() {
        // op0: reads reg0, writes reg1 (model output)
        let model = make_model(&[(&[0], &[1])], 2, &[1]);
        let liveness = analyze_liveness(&model);

        let all_freed: Vec<usize> = liveness.freed_after.iter().flatten().copied().collect();
        assert!(
            !all_freed.contains(&1),
            "model output reg1 must not be freed"
        );
    }

    /// Weight registers are never freed (they survive across inference runs).
    #[test]
    fn test_weight_registers_never_freed() {
        use crate::dispatch::WeightRegister;
        use crate::types::DataType;

        // op0: reads reg0 (weight) and reg1 (input), writes reg2 (output).
        // reg0 is a weight register — it must not appear in freed_after.
        let mut model = make_model(&[(&[0, 1], &[2])], 3, &[2]);
        model.weight_registers.push(WeightRegister {
            register: 0,
            data: vec![0u8; 16],
            shape: vec![4],
            dtype: DataType::F32,
        });
        let liveness = analyze_liveness(&model);

        let all_freed: Vec<usize> = liveness.freed_after.iter().flatten().copied().collect();
        assert!(!all_freed.contains(&0), "weight reg0 must never be freed");
        // reg1 (user input, last read by op0) CAN be freed
        assert!(
            liveness.freed_after[0].contains(&1),
            "input reg1 freed after op0"
        );
    }

    /// Empty model produces empty liveness info.
    #[test]
    fn test_empty_model() {
        let model = make_model(&[], 0, &[]);
        let liveness = analyze_liveness(&model);
        assert!(liveness.freed_after.is_empty());
    }
}
