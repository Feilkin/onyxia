//! Conditional operators.

use onyxia_core::{
    DataType, FoldCtx, InferenceCtx, Operator, PlanCtx, Result, Step, TensorData, TensorShape,
    TensorValue,
};

/// Where operator - selects elements from two tensors based on a condition.
///
/// Performs output = condition ? X : Y with broadcasting support.
pub struct WhereOp;

impl Operator for WhereOp {
    fn name(&self) -> &str {
        "Where"
    }

    fn infer_output_shapes(&self, ctx: &InferenceCtx) -> Result<Vec<TensorShape>> {
        // Where uses 3-way elementwise broadcasting
        crate::helpers::infer_elementwise_broadcast(ctx)
    }

    fn try_fold(&self, ctx: &FoldCtx) -> Result<Vec<Option<TensorValue>>> {
        let Some(condition) = ctx.input_value(0) else {
            return Ok(vec![None]);
        };
        let Some(x) = ctx.input_value(1) else {
            return Ok(vec![None]);
        };
        let Some(y) = ctx.input_value(2) else {
            return Ok(vec![None]);
        };

        if condition.len() == x.len() && condition.len() == y.len() {
            match (&condition.data, &x.data, &y.data) {
                (TensorData::Bool(cond), TensorData::F32(x_vals), TensorData::F32(y_vals)) => {
                    let result: Vec<f32> = cond
                        .iter()
                        .zip(x_vals.iter())
                        .zip(y_vals.iter())
                        .map(|((c, x), y)| if *c { *x } else { *y })
                        .collect();
                    return Ok(vec![Some(TensorValue::new(
                        TensorData::F32(result),
                        x.shape.clone(),
                        DataType::F32,
                    ))]);
                }
                (TensorData::Bool(cond), TensorData::I64(x_vals), TensorData::I64(y_vals)) => {
                    let result: Vec<i64> = cond
                        .iter()
                        .zip(x_vals.iter())
                        .zip(y_vals.iter())
                        .map(|((c, x), y)| if *c { *x } else { *y })
                        .collect();
                    return Ok(vec![Some(TensorValue::new(
                        TensorData::I64(result),
                        x.shape.clone(),
                        DataType::I64,
                    ))]);
                }
                _ => {}
            }
        }

        Ok(vec![None])
    }

    fn plan(&self, ctx: &mut PlanCtx) -> Result<Vec<Step>> {
        let output_tensor = ctx.output_tensor(0)?;
        let output_shape = ctx.static_dims(&output_tensor.shape)?;
        let num_elements: usize = output_shape.iter().product();

        let workgroup_size: u32 = 256;
        let num_workgroups = (num_elements as u32).div_ceil(workgroup_size);

        let mut shader_defs = std::collections::HashMap::new();
        shader_defs.insert("WORKGROUP_SIZE".to_string(), workgroup_size.to_string());

        let shader_index = ctx.compile_shader(
            "where",
            include_str!("../../shaders/elementwise/where.wgsl"),
            &shader_defs,
        )?;

        let input_condition = ctx.input_tensor(0)?;
        let condition_shape = ctx.static_dims(&input_condition.shape)?;
        let condition_size: u32 = condition_shape.iter().product::<usize>() as u32;

        let input_x = ctx.input_tensor(1)?;
        let x_shape = ctx.static_dims(&input_x.shape)?;
        let x_size: u32 = x_shape.iter().product::<usize>() as u32;

        let input_y = ctx.input_tensor(2)?;
        let y_shape = ctx.static_dims(&input_y.shape)?;
        let y_size: u32 = y_shape.iter().product::<usize>() as u32;

        let mut immediates_data = Vec::new();
        immediates_data.extend_from_slice(&(num_elements as u32).to_le_bytes());
        immediates_data.extend_from_slice(&condition_size.to_le_bytes());
        immediates_data.extend_from_slice(&x_size.to_le_bytes());
        immediates_data.extend_from_slice(&y_size.to_le_bytes());

        Ok(vec![Step::Dispatch {
            shader_index,
            bindings: vec![
                onyxia_core::BindingDesc {
                    buffer: ctx.input(0)?,
                    read_only: true,
                },
                onyxia_core::BindingDesc {
                    buffer: ctx.input(1)?,
                    read_only: true,
                },
                onyxia_core::BindingDesc {
                    buffer: ctx.input(2)?,
                    read_only: true,
                },
                onyxia_core::BindingDesc {
                    buffer: ctx.output(0)?,
                    read_only: false,
                },
            ],
            workgroups: [num_workgroups, 1, 1],
            immediates: Some(immediates_data),
        }])
    }
}
