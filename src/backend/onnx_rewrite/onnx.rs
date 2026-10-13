//! ONNX messages, only what the rewrite reads: the field numbers of `ModelProto`, `GraphProto`,
//! `NodeProto`, `AttributeProto`, `TensorProto` and `ValueInfoProto`, and lazily parsed views
//! (`Attr`, `Node`, `Tensor`, `ValueType`) that borrow the encoded model. Large repeated fields
//! (weights, `ints`) are scanned on demand rather than stored per value (BE-06).

use super::proto::*;
use anyhow::Result;

// ModelProto
pub(super) const MODEL_GRAPH: u32 = 7;
pub(super) const MODEL_OPSET_IMPORT: u32 = 8;
pub(super) const MODEL_METADATA_PROPS: u32 = 14;
// OperatorSetIdProto
pub(super) const OPSET_DOMAIN: u32 = 1;
pub(super) const OPSET_VERSION: u32 = 2;
// GraphProto
pub(super) const GRAPH_NODE: u32 = 1;
pub(super) const GRAPH_INITIALIZER: u32 = 5;
pub(super) const GRAPH_INPUT: u32 = 11;
pub(super) const GRAPH_OUTPUT: u32 = 12;
pub(super) const GRAPH_VALUE_INFO: u32 = 13;
// NodeProto
pub(super) const NODE_INPUT: u32 = 1;
pub(super) const NODE_OUTPUT: u32 = 2;
pub(super) const NODE_NAME: u32 = 3;
pub(super) const NODE_OP_TYPE: u32 = 4;
pub(super) const NODE_ATTRIBUTE: u32 = 5;
pub(super) const NODE_DOMAIN: u32 = 7;
// AttributeProto
pub(super) const ATTR_NAME: u32 = 1;
pub(super) const ATTR_F: u32 = 2;
pub(super) const ATTR_I: u32 = 3;
pub(super) const ATTR_T: u32 = 5;
pub(super) const ATTR_G: u32 = 6;
pub(super) const ATTR_INTS: u32 = 8;
pub(super) const ATTR_GRAPHS: u32 = 11;
pub(super) const ATTR_TYPE: u32 = 20;
pub(super) const ATTR_TYPE_FLOAT: i64 = 1;
pub(super) const ATTR_TYPE_INTS: i64 = 7;
// TensorProto
pub(super) const TENSOR_DIMS: u32 = 1;
pub(super) const TENSOR_DATA_TYPE: u32 = 2;
pub(super) const TENSOR_INT32_DATA: u32 = 5;
pub(super) const TENSOR_INT64_DATA: u32 = 7;
pub(super) const TENSOR_NAME: u32 = 8;
pub(super) const TENSOR_RAW_DATA: u32 = 9;
pub(super) const TENSOR_EXTERNAL_DATA: u32 = 13;
pub(super) const TENSOR_DATA_LOCATION: u32 = 14;
// ValueInfoProto / TypeProto / TensorShapeProto
pub(super) const VI_NAME: u32 = 1;
pub(super) const VI_TYPE: u32 = 2;
pub(super) const TYPE_TENSOR: u32 = 1;
pub(super) const TT_ELEM_TYPE: u32 = 1;
pub(super) const TT_SHAPE: u32 = 2;
pub(super) const SHAPE_DIM: u32 = 1;
pub(super) const DIM_VALUE: u32 = 1;
// StringStringEntryProto
pub(super) const ENTRY_KEY: u32 = 1;
pub(super) const ENTRY_VALUE: u32 = 2;

// TensorProto.DataType
pub const DT_FLOAT: i32 = 1;
pub const DT_INT32: i32 = 6;
pub const DT_INT64: i32 = 7;
pub const DT_FLOAT16: i32 = 10;
pub const DT_DOUBLE: i32 = 11;

pub(super) fn is_float_type(t: i32) -> bool {
    matches!(t, 1 | 10 | 11 | 16..=20 | 23)
}

/// The string in the last field of a scanned message's field number (protobuf "last one wins";
/// "" when that field is absent or not length-delimited, as [`get_str`]).
pub(super) fn str_of(last: Option<Value<'_>>) -> Result<&str> {
    match last {
        Some(Value::Len(b)) => as_str(b),
        _ => Ok(""),
    }
}

pub(super) fn bytes_of(last: Option<Value<'_>>) -> Option<&[u8]> {
    match last {
        Some(Value::Len(b)) => Some(b),
        _ => None,
    }
}

/// An `AttributeProto`, scanned rather than split into stored fields: an attribute may carry
/// large repeated values (`floats`, `ints`, a tensor), so `ints` is only decoded on demand
/// (BE-06).
#[derive(Debug)]
pub(super) struct Attr<'a> {
    pub(super) name: &'a str,
    pub(super) f: Option<f32>,
    pub(super) i: Option<i64>,
    pub(super) t: Option<&'a [u8]>,
    pub(super) has_graph: bool,
    /// The encoded message (for [`Attr::ints`]).
    pub(super) raw: &'a [u8],
}

impl Attr<'_> {
    /// The `ints` values (validated by [`parse_attr`]).
    pub(super) fn ints(&self) -> Vec<i64> {
        collect_ints(self.raw, ATTR_INTS).unwrap_or_default()
    }
}

/// Same results and errors as splitting the message with [`parse_fields`] and reading the last
/// occurrence of each singular field (protobuf "last one wins"), without storing a field per
/// value.
pub(super) fn parse_attr(b: &[u8]) -> Result<Attr<'_>> {
    let (mut name, mut f, mut i, mut t, mut has_graph) = (None, None, None, None, false);
    scan(b, |x| {
        match (x.num, x.value) {
            (ATTR_NAME, v) => name = Some(v),
            (ATTR_F, Value::Fixed32(v)) => f = Some(f32::from_bits(v)),
            (ATTR_I, Value::Varint(v)) => i = Some(v as i64),
            (ATTR_T, v) => t = Some(v),
            (ATTR_G | ATTR_GRAPHS, _) => has_graph = true,
            _ => {}
        }
        Ok(())
    })?;
    let name = str_of(name)?;
    // Malformed `ints` fail the parse here, as the eager decoding did.
    scan_ints(b, ATTR_INTS, |_| {})?;
    Ok(Attr {
        name,
        f,
        i,
        t: bytes_of(t),
        has_graph,
        raw: b,
    })
}

#[derive(Debug)]
pub(super) struct Node<'a> {
    pub(super) inputs: Vec<&'a str>,
    pub(super) outputs: Vec<&'a str>,
    pub(super) name: &'a str,
    pub(super) op_type: &'a str,
    pub(super) domain: &'a str,
    pub(super) attrs: Vec<Attr<'a>>,
}

impl<'a> Node<'a> {
    pub(super) fn parse(b: &'a [u8]) -> Result<Self> {
        let f = parse_fields(b)?;
        let attrs = f
            .iter()
            .filter(|x| x.num == NODE_ATTRIBUTE)
            .filter_map(len_of)
            .map(parse_attr)
            .collect::<Result<_>>()?;
        Ok(Self {
            inputs: rep_str(&f, NODE_INPUT)?,
            outputs: rep_str(&f, NODE_OUTPUT)?,
            name: get_str(&f, NODE_NAME)?,
            op_type: get_str(&f, NODE_OP_TYPE)?,
            domain: get_str(&f, NODE_DOMAIN)?,
            attrs,
        })
    }
    pub(super) fn attr(&self, name: &str) -> Option<&Attr<'a>> {
        self.attrs.iter().rev().find(|a| a.name == name)
    }
    pub(super) fn is_onnx(&self) -> bool {
        self.domain.is_empty() || self.domain == "ai.onnx"
    }
    /// Input `i` when present and not the empty "optional input omitted" name.
    pub(super) fn input(&self, i: usize) -> Option<&'a str> {
        self.inputs.get(i).copied().filter(|s| !s.is_empty())
    }
}

/// A `TensorProto` (initializer or attribute tensor). Weights make up nearly all of a model, so
/// the message is scanned once for the few scalar fields the rewrite needs and otherwise kept as
/// its encoded bytes; dims and integer data are decoded on demand (BE-06). The tensor itself is
/// always copied through unchanged.
#[derive(Debug)]
pub(super) struct Tensor<'a> {
    pub(super) name: &'a str,
    pub(super) data_type: i32,
    pub(super) external: bool,
    /// Last `raw_data` field, if length-delimited.
    pub(super) raw_data: Option<&'a [u8]>,
    /// The encoded message.
    pub(super) raw: &'a [u8],
}

impl<'a> Tensor<'a> {
    /// Same results and errors as splitting the message with [`parse_fields`] and reading the
    /// last occurrence of each singular field, without storing a field per value.
    pub(super) fn parse(b: &'a [u8]) -> Result<Self> {
        let (mut name, mut data_type, mut location, mut raw_data, mut ext) =
            (None, None, None, None, false);
        scan(b, |f| {
            match (f.num, f.value) {
                (TENSOR_NAME, v) => name = Some(v),
                (TENSOR_DATA_TYPE, Value::Varint(v)) => data_type = Some(v as i64),
                (TENSOR_DATA_LOCATION, Value::Varint(v)) => location = Some(v as i64),
                (TENSOR_RAW_DATA, v) => raw_data = Some(v),
                (TENSOR_EXTERNAL_DATA, _) => ext = true,
                _ => {}
            }
            Ok(())
        })?;
        let name = str_of(name)?;
        // Malformed dims fail the parse here, as the eager decoding did.
        scan_ints(b, TENSOR_DIMS, |_| {})?;
        Ok(Self {
            name,
            data_type: data_type.and_then(|v| i32::try_from(v).ok()).unwrap_or(0),
            external: location == Some(1) || ext,
            raw_data: bytes_of(raw_data),
            raw: b,
        })
    }

    /// The dimensions (validated by [`Tensor::parse`]).
    pub(super) fn dims(&self) -> Vec<i64> {
        collect_ints(self.raw, TENSOR_DIMS).unwrap_or_default()
    }

    /// Integer contents of an int64/int32 tensor (raw or typed data).
    pub(super) fn ints(&self) -> Option<Vec<i64>> {
        if self.external {
            return None;
        }
        let raw = self.raw_data;
        match self.data_type {
            DT_INT64 => match raw {
                Some(r) if r.len() % 8 == 0 => Some(
                    r.as_chunks::<8>()
                        .0
                        .iter()
                        .map(|c| i64::from_le_bytes(*c))
                        .collect(),
                ),
                Some(_) => None,
                None => collect_ints(self.raw, TENSOR_INT64_DATA).ok(),
            },
            DT_INT32 => match raw {
                Some(r) if r.len() % 4 == 0 => Some(
                    r.as_chunks::<4>()
                        .0
                        .iter()
                        .map(|c| i64::from(i32::from_le_bytes(*c)))
                        .collect(),
                ),
                Some(_) => None,
                None => collect_ints(self.raw, TENSOR_INT32_DATA)
                    .ok()?
                    .into_iter()
                    .map(|x| i32::try_from(x).ok().map(i64::from))
                    .collect(),
            },
            _ => None,
        }
    }
}

/// Element type and (partial) shape of a value from `ValueInfoProto`.
#[derive(Debug, Clone, Default, PartialEq)]
pub(super) struct ValueType {
    pub(super) elem: Option<i32>,
    /// None = rank unknown; `Some(None)` dims are symbolic/unknown.
    pub(super) shape: Option<Vec<Option<i64>>>,
}

pub(super) fn parse_value_info(b: &[u8]) -> Result<(&str, ValueType)> {
    let f = parse_fields(b)?;
    let name = get_str(&f, VI_NAME)?;
    let mut vt = ValueType::default();
    if let Some(tp) = get_len(&f, VI_TYPE) {
        let tp = parse_fields(tp)?;
        if let Some(tt) = get_len(&tp, TYPE_TENSOR) {
            let tt = parse_fields(tt)?;
            vt.elem = get_int(&tt, TT_ELEM_TYPE).and_then(|v| i32::try_from(v).ok());
            if let Some(shape) = get_len(&tt, TT_SHAPE) {
                let shape = parse_fields(shape)?;
                let mut dims = Vec::new();
                for d in shape
                    .iter()
                    .filter(|x| x.num == SHAPE_DIM)
                    .filter_map(len_of)
                {
                    let d = parse_fields(d)?;
                    dims.push(get_int(&d, DIM_VALUE).filter(|v| *v >= 0));
                }
                vt.shape = Some(dims);
            }
        }
    }
    Ok((name, vt))
}

/// `ai.onnx` opset of a parsed ModelProto (None when the model imports none).
pub(super) fn onnx_opset(model: &[Field<'_>]) -> Result<Option<i64>> {
    let mut opset = None;
    for b in model
        .iter()
        .filter(|f| f.num == MODEL_OPSET_IMPORT)
        .filter_map(len_of)
    {
        let f = parse_fields(b)?;
        let domain = get_str(&f, OPSET_DOMAIN)?;
        if domain.is_empty() || domain == "ai.onnx" {
            opset = get_int(&f, OPSET_VERSION);
        }
    }
    Ok(opset)
}

#[cfg(test)]
mod tests {
    use super::super::RewriteOptions;
    use super::super::rewrite;
    use super::super::test_models::*;
    use super::*;

    /// The lazy tensor/attribute parse keeps the eager parse's protobuf semantics: last
    /// occurrence wins (and only with the expected wire type for numbers), and malformed
    /// repeated integers still fail the parse.
    #[test]
    fn lazy_parse_matches_eager_semantics() {
        let t = msg(|w| {
            w.string(TENSOR_NAME, "first");
            w.int(TENSOR_DATA_TYPE, i64::from(DT_INT64));
            w.float(TENSOR_DATA_TYPE, 1.0); // wrong wire type: ignored, as get_int did
            w.int(TENSOR_DIMS, 2);
            w.bytes(TENSOR_RAW_DATA, &[0; 16]);
            w.string(TENSOR_NAME, "last");
            w.bytes(
                TENSOR_RAW_DATA,
                &[1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
            );
        });
        let p = Tensor::parse(&t).unwrap();
        assert_eq!((p.name, p.data_type, p.dims()), ("last", DT_INT64, vec![2]));
        assert_eq!(p.ints(), Some(vec![1, 2]));
        // A trailing non-bytes name field hides the earlier one (get_str semantics).
        let t2 = msg(|w| {
            w.string(TENSOR_NAME, "n");
            w.int(TENSOR_NAME, 3);
        });
        assert_eq!(Tensor::parse(&t2).unwrap().name, "");
        let bad_dims = msg(|w| w.float(TENSOR_DIMS, 1.0));
        assert!(Tensor::parse(&bad_dims).is_err());
        let bad_name = msg(|w| w.bytes(TENSOR_NAME, &[0xff]));
        assert!(Tensor::parse(&bad_name).is_err());
        let bad_ints = msg(|w| {
            w.string(ATTR_NAME, "split");
            w.bytes(ATTR_INTS, &[0x80]); // truncated packed varint
        });
        assert!(parse_attr(&bad_ints).is_err());
        let a = attr_ints("split", &[1, -2, 3], false);
        let mut a2 = a.clone();
        a2.extend(msg(|w| w.bytes(ATTR_INTS, &msg(|p| p.varint(4)))));
        assert_eq!(parse_attr(&a2).unwrap().ints(), [1, -2, 3, 4]);
        // The same model fails the rewrite as a whole, as before.
        let sp = node("Split", "s", &["x"], &["a", "b"], &[bad_ints]);
        assert!(rewrite(&single(12, sp, &["a", "b"], &[2]), &RewriteOptions::COREML).is_err());
    }
}
