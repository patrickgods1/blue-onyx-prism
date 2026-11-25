//! Output-reading helpers shared by every model family: find an output by name, the primary
//! f32 output of a YOLO-style head, `[1, R, C]` matrix dims checked against the buffer, and the
//! NaN-safe score threshold test. Their error messages name the calling family.

use anyhow::Result;

use super::NamedOutput;

/// The output called `name`, or an error naming `family` and listing what the model has.
pub(crate) fn find_output<'a>(
    family: &str,
    outputs: &'a [NamedOutput],
    name: &str,
) -> Result<&'a NamedOutput> {
    outputs.iter().find(|o| o.name == name).ok_or_else(|| {
        anyhow::anyhow!(
            "{family}: missing output `{name}`; got {:?}",
            outputs
                .iter()
                .map(|o| (&o.name, &o.shape))
                .collect::<Vec<_>>()
        )
    })
}

/// The single detection output of a YOLO-style model, as f32 data. When several outputs are
/// present, the first f32 one is used (some exports append auxiliary heads after the main one).
pub(crate) fn primary_f32_output<'a>(
    family: &str,
    outputs: &'a [NamedOutput],
) -> Result<(&'a NamedOutput, &'a [f32])> {
    let out = outputs
        .iter()
        .find(|o| o.as_f32().is_some())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "{family}: expected an f32 output tensor, got {:?}",
                outputs
                    .iter()
                    .map(|o| (&o.name, &o.shape))
                    .collect::<Vec<_>>()
            )
        })?;
    let data = out.as_f32().unwrap_or_default();
    Ok((out, data))
}

/// `value >= threshold`; false for NaN scores.
#[inline]
pub(crate) fn meets(value: f32, threshold: f32) -> bool {
    value >= threshold
}

/// Interpret `[1, R, C]` or `[R, C]` as `(R, C)` and check the data length.
pub(crate) fn matrix_dims(
    family: &str,
    expected: &str,
    out: &NamedOutput,
    len: usize,
) -> Result<(usize, usize)> {
    let dims = match out.shape.as_slice() {
        [1, r, c] | [r, c] => (*r, *c),
        _ => anyhow::bail!(
            "{family}: output `{}` has shape {:?}, expected {expected}",
            out.name,
            out.shape
        ),
    };
    let implied = dims.0.checked_mul(dims.1);
    if implied != Some(len) {
        anyhow::bail!(
            "{family}: output `{}` shape {:?} implies {} values but the buffer has {len}",
            out.name,
            out.shape,
            implied.map_or_else(|| "too many".to_string(), |n| n.to_string())
        );
    }
    Ok(dims)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::OutputBuf;

    fn out(name: &str, shape: &[usize], data: OutputBuf) -> NamedOutput {
        NamedOutput {
            name: name.into(),
            shape: shape.to_vec(),
            data,
        }
    }

    #[test]
    fn find_output_names_the_family_and_lists_the_outputs() {
        let outs = [out("boxes", &[1, 2, 4], OutputBuf::F32(vec![0.0; 8]))];
        assert_eq!(find_output("detr", &outs, "boxes").unwrap().name, "boxes");
        let err = find_output("rtdetr", &outs, "labels").unwrap_err();
        assert_eq!(
            err.to_string(),
            "rtdetr: missing output `labels`; got [(\"boxes\", [1, 2, 4])]"
        );
    }

    #[test]
    fn primary_output_is_the_first_f32_one() {
        let outs = [
            out("idx", &[1, 2], OutputBuf::I64(vec![1, 2])),
            out("a", &[1, 1], OutputBuf::F32(vec![0.5])),
            out("b", &[1, 1], OutputBuf::F32(vec![0.7])),
        ];
        let (o, data) = primary_f32_output("yolo5", &outs).unwrap();
        assert_eq!((o.name.as_str(), data), ("a", &[0.5][..]));
        let err = primary_f32_output("yolo8", &outs[..1]).unwrap_err();
        assert_eq!(
            err.to_string(),
            "yolo8: expected an f32 output tensor, got [(\"idx\", [1, 2])]"
        );
    }

    #[test]
    fn meets_is_false_for_nan() {
        assert!(meets(0.5, 0.5));
        assert!(!meets(0.49, 0.5));
        assert!(!meets(f32::NAN, 0.0));
    }

    #[test]
    fn matrix_dims_accepts_two_and_three_dims_and_checks_the_length() {
        let o = out("o", &[1, 3, 6], OutputBuf::F32(Vec::new()));
        assert_eq!(matrix_dims("yolo26", "[1, N, 6]", &o, 18).unwrap(), (3, 6));
        let o2 = out("o", &[3, 6], OutputBuf::F32(Vec::new()));
        assert_eq!(matrix_dims("yolo26", "[1, N, 6]", &o2, 18).unwrap(), (3, 6));
        assert_eq!(
            matrix_dims("yolo26", "[1, N, 6]", &o, 17)
                .unwrap_err()
                .to_string(),
            "yolo26: output `o` shape [1, 3, 6] implies 18 values but the buffer has 17"
        );
        let o4 = out("o", &[2, 1, 3, 6], OutputBuf::F32(Vec::new()));
        assert_eq!(
            matrix_dims("yolo5", "[1, N, 5+C]", &o4, 36)
                .unwrap_err()
                .to_string(),
            "yolo5: output `o` has shape [2, 1, 3, 6], expected [1, N, 5+C]"
        );
        let big = out("o", &[usize::MAX, 2], OutputBuf::F32(Vec::new()));
        assert!(
            matrix_dims("yolo8", "[1, 4+C, A]", &big, 4)
                .unwrap_err()
                .to_string()
                .contains("implies too many values")
        );
    }
}
