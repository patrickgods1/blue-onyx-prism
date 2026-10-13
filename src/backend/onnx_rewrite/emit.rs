//! Rewrite internals: the parsed main graph (`Graph`), unique-name and shared-initializer
//! minting (`Builder`), the replacement emitters (`emit_*`) and re-serialization (`assemble`).
//! Emitters return `Err(reason)` instead of failing the model: the node is then kept and listed
//! as skipped. `assemble` copies every untouched field byte for byte.

use super::onnx::*;
use super::proto::*;
use anyhow::{Context, Result};
use std::collections::{BTreeSet, HashMap, HashSet};

/// Everything known about the main graph.
pub(super) struct Graph<'a> {
    pub(super) fields: Vec<Field<'a>>,
    /// (field index, node)
    pub(super) nodes: Vec<(usize, Node<'a>)>,
    /// name -> (field index, tensor)
    pub(super) inits: HashMap<&'a str, (usize, Tensor<'a>)>,
    pub(super) types: HashMap<&'a str, ValueType>,
    pub(super) inputs: HashSet<&'a str>,
    pub(super) outputs: HashSet<&'a str>,
    pub(super) names: HashSet<String>,
    pub(super) has_subgraphs: bool,
    pub(super) external_data: bool,
    /// Element type assumed for float values whose type is not recorded: f32 when no other
    /// floating type occurs anywhere in the model.
    pub(super) default_float: Option<i32>,
}

impl<'a> Graph<'a> {
    pub(super) fn parse(b: &'a [u8]) -> Result<Self> {
        let fields = parse_fields(b)?;
        let mut g = Graph {
            fields: Vec::new(),
            nodes: Vec::new(),
            inits: HashMap::new(),
            types: HashMap::new(),
            inputs: HashSet::new(),
            outputs: HashSet::new(),
            names: HashSet::new(),
            has_subgraphs: false,
            external_data: false,
            default_float: None,
        };
        let mut float_types = BTreeSet::new();
        for (i, f) in fields.iter().enumerate() {
            let Some(b) = len_of(f) else { continue };
            match f.num {
                GRAPH_NODE => {
                    let n = Node::parse(b).with_context(|| format!("node #{}", g.nodes.len()))?;
                    for a in &n.attrs {
                        g.has_subgraphs |= a.has_graph;
                        if let Some(t) = a.t {
                            let t = Tensor::parse(t)?;
                            g.external_data |= t.external;
                            float_types.insert(t.data_type);
                        }
                    }
                    if matches!(n.op_type, "Cast" | "CastLike")
                        && let Some(to) = n.attr("to").and_then(|a| a.i)
                        && let Ok(to) = i32::try_from(to)
                    {
                        float_types.insert(to);
                    }
                    g.names.extend(n.inputs.iter().map(|s| s.to_string()));
                    g.names.extend(n.outputs.iter().map(|s| s.to_string()));
                    g.names.insert(n.name.to_string());
                    g.nodes.push((i, n));
                }
                GRAPH_INITIALIZER => {
                    let t = Tensor::parse(b)?;
                    g.external_data |= t.external;
                    float_types.insert(t.data_type);
                    g.names.insert(t.name.to_string());
                    g.inits.insert(t.name, (i, t));
                }
                GRAPH_INPUT | GRAPH_OUTPUT | GRAPH_VALUE_INFO => {
                    let (name, vt) = parse_value_info(b)?;
                    if let Some(e) = vt.elem {
                        float_types.insert(e);
                    }
                    g.names.insert(name.to_string());
                    match f.num {
                        GRAPH_INPUT => g.inputs.insert(name),
                        GRAPH_OUTPUT => g.outputs.insert(name),
                        _ => false,
                    };
                    g.types.entry(name).or_insert(vt);
                }
                _ => {}
            }
        }
        float_types.retain(|t| is_float_type(*t));
        g.default_float = match float_types.iter().collect::<Vec<_>>()[..] {
            [] | [&DT_FLOAT] => Some(DT_FLOAT),
            _ => None,
        };
        g.fields = fields;
        Ok(g)
    }

    pub(super) fn elem_type(&self, name: &str) -> Option<i32> {
        self.types
            .get(name)
            .and_then(|t| t.elem)
            .or_else(|| self.inits.get(name).map(|(_, t)| t.data_type))
            .or(self.default_float)
    }

    pub(super) fn shape(&self, name: &str) -> Option<Vec<Option<i64>>> {
        self.types
            .get(name)
            .and_then(|t| t.shape.clone())
            .or_else(|| {
                self.inits
                    .get(name)
                    .map(|(_, t)| t.dims().into_iter().map(Some).collect())
            })
    }

    /// Integer contents of an initializer or `Constant` output.
    pub(super) fn const_ints(&self, name: &str) -> Option<Vec<i64>> {
        if let Some((_, t)) = self.inits.get(name) {
            return t.ints();
        }
        let (_, n) = self
            .nodes
            .iter()
            .find(|(_, n)| n.op_type == "Constant" && n.outputs.first() == Some(&name))?;
        if let Some(a) = n.attr("value_ints") {
            return Some(a.ints());
        }
        if let Some(a) = n.attr("value_int") {
            return a.i.map(|v| vec![v]);
        }
        Tensor::parse(n.attr("value")?.t?).ok()?.ints()
    }
}

/// Encodes new nodes and initializers with names unique in the graph.
pub(super) struct Builder<'g> {
    pub(super) names: &'g mut HashSet<String>,
    pub(super) inits: Vec<Vec<u8>>,
    pub(super) consts: HashMap<(i32, Vec<i64>, Vec<u8>), String>,
}

/// Attribute values the rewrite emits.
pub(super) enum AttrVal {
    F(f32),
    Ints(Vec<i64>),
}

impl Builder<'_> {
    pub(super) fn fresh(&mut self, base: &str) -> String {
        let mut name = base.to_string();
        let mut k = 1;
        while self.names.contains(&name) {
            name = format!("{base}_{k}");
            k += 1;
        }
        self.names.insert(name.clone());
        name
    }

    /// A shared initializer (one per distinct type/shape/value).
    pub(super) fn constant(&mut self, data_type: i32, dims: &[i64], raw: Vec<u8>) -> String {
        let key = (data_type, dims.to_vec(), raw);
        if let Some(n) = self.consts.get(&key) {
            return n.clone();
        }
        let name = self.fresh(&format!("bop_rewrite/const_{}", self.consts.len()));
        let mut w = Writer::default();
        for d in dims {
            w.int(TENSOR_DIMS, *d);
        }
        w.int(TENSOR_DATA_TYPE, i64::from(data_type));
        w.string(TENSOR_NAME, &name);
        w.bytes(TENSOR_RAW_DATA, &key.2);
        self.inits.push(w.buf);
        self.consts.insert(key, name.clone());
        name
    }

    /// Scalar of a floating element type (f32, f16, f64); None for other types.
    pub(super) fn scalar(&mut self, data_type: i32, v: f32) -> Option<String> {
        let raw = match data_type {
            DT_FLOAT => v.to_le_bytes().to_vec(),
            DT_DOUBLE => f64::from(v).to_le_bytes().to_vec(),
            DT_FLOAT16 => f32_to_f16_bits(v).to_le_bytes().to_vec(),
            _ => return None,
        };
        Some(self.constant(data_type, &[], raw))
    }

    pub(super) fn ints(&mut self, v: &[i64]) -> String {
        let raw = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        self.constant(DT_INT64, &[v.len() as i64], raw)
    }
}

fn encode_node(
    inputs: &[&str],
    outputs: &[&str],
    name: &str,
    op: &str,
    attrs: &[(&str, AttrVal)],
) -> Vec<u8> {
    let mut w = Writer::default();
    for i in inputs {
        w.string(NODE_INPUT, i);
    }
    for o in outputs {
        w.string(NODE_OUTPUT, o);
    }
    w.string(NODE_NAME, name);
    w.string(NODE_OP_TYPE, op);
    for (n, v) in attrs {
        let mut a = Writer::default();
        a.string(ATTR_NAME, n);
        match v {
            AttrVal::F(f) => {
                a.float(ATTR_F, *f);
                a.int(ATTR_TYPE, ATTR_TYPE_FLOAT);
            }
            AttrVal::Ints(v) => {
                for x in v {
                    a.int(ATTR_INTS, *x);
                }
                a.int(ATTR_TYPE, ATTR_TYPE_INTS);
            }
        }
        w.bytes(NODE_ATTRIBUTE, &a.buf);
    }
    w.buf
}

/// IEEE 754 binary16 bits of `x`, rounded to nearest even.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "e is in 1..=30 where cast to u32; r is at most 0x7c00 (+1) and sign is 0 or 0x8000, both fit u16"
)]
pub(super) fn f32_to_f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let man = b & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = man | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = 1u32 << (shift - 1);
        let rem = m & ((1u32 << shift) - 1);
        let mut r = m >> shift;
        if rem > half || (rem == half && r & 1 == 1) {
            r += 1;
        }
        return sign | r as u16;
    }
    let mut r = ((e as u32) << 10) | (man >> 13);
    let rem = man & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && r & 1 == 1) {
        r += 1; // may carry into the exponent (up to infinity), which is the correct rounding
    }
    sign | r as u16
}

/// Per-node context for emitting replacement nodes.
pub(super) struct Emit<'b, 'g> {
    pub(super) b: &'b mut Builder<'g>,
    pub(super) base: String,
    pub(super) nodes: Vec<Vec<u8>>,
}

impl Emit<'_, '_> {
    pub(super) fn node(&mut self, inputs: &[&str], outputs: &[&str], suffix: &str, op: &str) {
        self.node_with(inputs, outputs, suffix, op, &[]);
    }
    pub(super) fn node_with(
        &mut self,
        inputs: &[&str],
        outputs: &[&str],
        suffix: &str,
        op: &str,
        attrs: &[(&str, AttrVal)],
    ) {
        let name = self.b.fresh(&format!("{}/bop_{suffix}", self.base));
        self.nodes
            .push(encode_node(inputs, outputs, &name, op, attrs));
    }
}

/// `y = Clip(Add(Mul(x, alpha), beta), 0, 1)`; HardSwish multiplies by `x` on top.
pub(super) fn emit_hard_sigmoid(
    e: &mut Emit<'_, '_>,
    opset: i64,
    dtype: i32,
    x: &str,
    (alpha, beta): (f32, f32),
    y: &str,
    swish: bool,
) -> Result<(), String> {
    let (Some(a), Some(b)) = (e.b.scalar(dtype, alpha), e.b.scalar(dtype, beta)) else {
        return Err(format!("element type {dtype} not supported"));
    };
    let mul = e.b.fresh(&format!("{y}/bop_mul"));
    let add = e.b.fresh(&format!("{y}/bop_add"));
    let clip = if swish {
        e.b.fresh(&format!("{y}/bop_clip"))
    } else {
        y.to_string()
    };
    e.node(&[x, &a], &[&mul], "mul", "Mul");
    e.node(&[&mul, &b], &[&add], "add", "Add");
    if opset >= 11 {
        let (Some(lo), Some(hi)) = (e.b.scalar(dtype, 0.0), e.b.scalar(dtype, 1.0)) else {
            return Err(format!("element type {dtype} not supported"));
        };
        e.node(&[&add, &lo, &hi], &[&clip], "clip", "Clip");
    } else {
        e.node_with(
            &[&add],
            &[&clip],
            "clip",
            "Clip",
            &[("min", AttrVal::F(0.0)), ("max", AttrVal::F(1.0))],
        );
    }
    if swish {
        e.node(&[x, &clip], &[y], "mul_x", "Mul");
    }
    Ok(())
}

/// Split sizes along `axis` of a `Split` node, or why they are unknown.
pub(super) fn split_sizes(
    g: &Graph<'_>,
    n: &Node<'_>,
    opset: i64,
    axis: i64,
) -> Result<Vec<i64>, String> {
    let outs = i64::try_from(n.outputs.len()).map_err(|_| "too many outputs".to_string())?;
    let explicit = if opset < 13 {
        n.attr("split").map(Attr::ints)
    } else if let Some(s) = n.input(1) {
        Some(
            g.const_ints(s)
                .ok_or_else(|| format!("split sizes '{s}' are not a constant"))?,
        )
    } else {
        None
    };
    if let Some(v) = explicit {
        return Ok(v);
    }
    let x = n.inputs.first().copied().unwrap_or_default();
    let dim = g
        .shape(x)
        .and_then(|s| {
            let r = i64::try_from(s.len()).ok()?;
            let a = if axis < 0 { axis.checked_add(r)? } else { axis };
            s.get(usize::try_from(a).ok()?).copied().flatten()
        })
        .ok_or_else(|| format!("equal split of '{x}' needs its shape along axis {axis}"))?;
    if opset >= 18 {
        let k = n
            .attr("num_outputs")
            .and_then(|a| a.i)
            .ok_or("neither 'split' nor 'num_outputs' given")?;
        if k <= 0 || k != outs {
            return Err(format!("num_outputs {k} != {outs} outputs"));
        }
        // Chunks of ceil(dim / k); the last one is smaller when uneven.
        let chunk = dim
            .checked_add(k - 1)
            .map(|t| t / k)
            .ok_or_else(|| format!("dimension {dim} overflows when split into {k} chunks"))?;
        let mut v = Vec::new();
        let mut left = dim;
        for _ in 0..k {
            v.push(chunk.min(left).max(0));
            left = left.saturating_sub(chunk);
        }
        return Ok(v);
    }
    if outs == 0 || dim % outs != 0 {
        return Err(format!("dimension {dim} does not split evenly into {outs}"));
    }
    Ok(vec![dim / outs; n.outputs.len()])
}

pub(super) fn emit_split(
    e: &mut Emit<'_, '_>,
    g: &Graph<'_>,
    n: &Node<'_>,
    opset: i64,
) -> Result<(), String> {
    let x = n.input(0).ok_or("no input")?;
    let axis = n.attr("axis").and_then(|a| a.i).unwrap_or(0);
    let sizes = split_sizes(g, n, opset, axis)?;
    if sizes.len() != n.outputs.len() || sizes.iter().any(|s| *s < 0) {
        return Err(format!(
            "{} split sizes {sizes:?} for {} outputs",
            sizes.len(),
            n.outputs.len()
        ));
    }
    let mut start = 0i64;
    for (j, (size, out)) in sizes.iter().zip(&n.outputs).enumerate() {
        let end = start
            .checked_add(*size)
            .ok_or_else(|| format!("split sizes {sizes:?} overflow i64"))?;
        if !out.is_empty() {
            let suffix = format!("slice{j}");
            if opset >= 10 {
                let s = e.b.ints(&[start]);
                let en = e.b.ints(&[end]);
                let ax = e.b.ints(&[axis]);
                e.node(&[x, &s, &en, &ax], &[out], &suffix, "Slice");
            } else {
                e.node_with(
                    &[x],
                    &[out],
                    &suffix,
                    "Slice",
                    &[
                        ("starts", AttrVal::Ints(vec![start])),
                        ("ends", AttrVal::Ints(vec![end])),
                        ("axes", AttrVal::Ints(vec![axis])),
                    ],
                );
            }
        }
        start = end;
    }
    Ok(())
}

pub(super) fn emit_flatten(
    e: &mut Emit<'_, '_>,
    g: &Graph<'_>,
    n: &Node<'_>,
) -> Result<(), String> {
    let x = n.input(0).ok_or("no input")?;
    let y = n.outputs.first().copied().ok_or("no output")?;
    let mut axis = n.attr("axis").and_then(|a| a.i).unwrap_or(1);
    let shape = g.shape(x);
    if axis < 0 {
        let r = shape
            .as_ref()
            .ok_or("negative axis needs the input rank")?
            .len();
        axis = i64::try_from(r)
            .ok()
            .and_then(|r| axis.checked_add(r))
            .ok_or_else(|| format!("axis {axis} out of range"))?;
    }
    let target = match axis {
        0 => vec![1, -1],
        // Reshape copies dimension 0 for a 0 entry (allowzero = 0).
        1 => vec![0, -1],
        a if a > 1 => {
            let dims = shape.ok_or("axis > 1 needs the input shape")?;
            let lead: Option<Vec<i64>> = usize::try_from(a)
                .ok()
                .and_then(|a| dims.get(..a))
                .ok_or("axis beyond the input rank")?
                .iter()
                .copied()
                .collect();
            let lead = lead.ok_or("axis > 1 needs static leading dimensions")?;
            let product = lead
                .iter()
                .try_fold(1i64, |acc, d| acc.checked_mul(*d))
                .ok_or_else(|| format!("leading dimensions {lead:?} overflow i64"))?;
            vec![product, -1]
        }
        _ => return Err(format!("axis {axis} out of range")),
    };
    let s = e.b.ints(&target);
    e.node(&[x, &s], &[y], "reshape", "Reshape");
    Ok(())
}

/// Serialize the model again: untouched fields byte for byte, `replaced` graph fields (by index)
/// by their new nodes, `dropped` graph fields left out, `new_inits` appended to the graph and a
/// `marker` metadata entry appended to the model.
pub(super) fn assemble(
    mfields: &[Field<'_>],
    graph_index: usize,
    gfields: &[Field<'_>],
    replaced: &HashMap<usize, Vec<Vec<u8>>>,
    dropped: &HashSet<usize>,
    new_inits: &[Vec<u8>],
    marker: Option<&str>,
) -> Vec<u8> {
    let mut gw = Writer::default();
    for (i, f) in gfields.iter().enumerate() {
        if dropped.contains(&i) {
            continue;
        }
        match replaced.get(&i) {
            Some(nodes) => {
                for n in nodes {
                    gw.bytes(GRAPH_NODE, n);
                }
            }
            None => gw.raw(f.raw),
        }
    }
    for t in new_inits {
        gw.bytes(GRAPH_INITIALIZER, t);
    }
    let mut mw = Writer::default();
    for (i, f) in mfields.iter().enumerate() {
        if i == graph_index {
            mw.bytes(MODEL_GRAPH, &gw.buf);
        } else {
            mw.raw(f.raw);
        }
    }
    if let Some(m) = marker {
        let mut entry = Writer::default();
        entry.string(ENTRY_KEY, "blue_onyx_prism.rewrite");
        entry.string(ENTRY_VALUE, m);
        mw.bytes(MODEL_METADATA_PROPS, &entry.buf);
    }
    mw.buf
}

/// Parse `model` and serialize it again without changes; byte-identical to the input for any
/// model with minimal varints (what every protobuf encoder writes). Checks the wire-format layer.
pub fn reencode(model: &[u8]) -> Result<Vec<u8>> {
    let mfields = parse_fields(model)?;
    let gi = mfields
        .iter()
        .rposition(|f| f.num == MODEL_GRAPH)
        .context("model has no graph")?;
    let gfields = parse_fields(mfields.get(gi).and_then(len_of).context("graph field")?)?;
    Ok(assemble(
        &mfields,
        gi,
        &gfields,
        &HashMap::new(),
        &HashSet::new(),
        &[],
        None,
    ))
}

#[cfg(test)]
mod tests {
    use super::super::test_models::*;
    use super::super::{Outcome, RewriteOptions, rewrite};
    use super::*;

    #[test]
    fn f16_conversion() {
        assert_eq!(f32_to_f16_bits(0.0), 0x0000);
        assert_eq!(f32_to_f16_bits(-0.0), 0x8000);
        assert_eq!(f32_to_f16_bits(1.0), 0x3c00);
        assert_eq!(f32_to_f16_bits(0.5), 0x3800);
        assert_eq!(f32_to_f16_bits(1.0 / 6.0), 0x3155);
        assert_eq!(f32_to_f16_bits(0.2), 0x3266);
        assert_eq!(f32_to_f16_bits(-2.0), 0xc000);
        assert_eq!(f32_to_f16_bits(65504.0), 0x7bff);
        assert_eq!(f32_to_f16_bits(65520.0), 0x7c00, "rounds up to infinity");
        assert_eq!(f32_to_f16_bits(f32::INFINITY), 0x7c00);
        assert_eq!(f32_to_f16_bits(f32::NAN) & 0x7e00, 0x7e00);
        assert_eq!(
            f32_to_f16_bits(2f32.powi(-24)),
            0x0001,
            "smallest subnormal"
        );
        assert_eq!(f32_to_f16_bits(2f32.powi(-14)), 0x0400, "smallest normal");
        assert_eq!(f32_to_f16_bits(2f32.powi(-26)), 0x0000);
        // 1 + 2^-11 is halfway between 1 and the next f16: ties to even (1.0).
        assert_eq!(f32_to_f16_bits(1.0 + 2f32.powi(-11)), 0x3c00);
        assert_eq!(f32_to_f16_bits(1.0 + 3.0 * 2f32.powi(-11)), 0x3c02);
    }

    #[test]
    fn reencode_is_byte_identical() {
        let hs = node(
            "HardSigmoid",
            "hs",
            &["x"],
            &["y"],
            &[attr_f("alpha", 0.25)],
        );
        let mut unknown = Writer::default();
        unknown.string(999, "future field");
        unknown.int(1000, -5);
        unknown.float(1001, 1.5);
        unknown.key(1002, 1);
        unknown.raw(&7u64.to_le_bytes());
        let g = G {
            nodes: vec![hs],
            inputs: vec![value_info("x", DT_FLOAT, Some(&[1, 3, -1, 4]))],
            outputs: vec![value_info("y", DT_FLOAT, None)],
            extra: unknown.buf.clone(),
            ..G::default()
        };
        let m = model(12, &g, &unknown.buf);
        assert_eq!(reencode(&m).unwrap(), m);
        // Nothing to rewrite: unchanged, no copy.
        let opts = RewriteOptions {
            hard_sigmoid: false,
            ..RewriteOptions::ALL
        };
        assert!(matches!(
            rewrite(&m, &opts).unwrap(),
            Outcome::Unchanged { ref skipped } if skipped.is_empty()
        ));
    }
}
