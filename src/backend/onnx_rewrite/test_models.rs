//! Test-only builders of synthetic ONNX models (`model`, `single`, `node`, ...), a decoded
//! `View` of a rewritten model for assertions, the golden corpus covering every rewrite path,
//! and the `FIELDS_STORED` counter `parse_fields` bumps under `cfg(test)` (BE-06 memory test).

use super::onnx::*;
use super::proto::*;
use super::*;
use std::collections::HashMap;

thread_local! {
    /// Fields stored by [`parse_fields`] on this thread (BE-06 memory test).
    pub(super) static FIELDS_STORED: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

// -- synthetic ONNX models -------------------------------------------------------------

pub(super) fn msg(f: impl FnOnce(&mut Writer)) -> Vec<u8> {
    let mut w = Writer::default();
    f(&mut w);
    w.buf
}

pub(super) fn attr_f(name: &str, v: f32) -> Vec<u8> {
    msg(|w| {
        w.string(ATTR_NAME, name);
        w.float(ATTR_F, v);
        w.int(ATTR_TYPE, ATTR_TYPE_FLOAT);
    })
}

pub(super) fn attr_i(name: &str, v: i64) -> Vec<u8> {
    msg(|w| {
        w.string(ATTR_NAME, name);
        w.int(ATTR_I, v);
        w.int(ATTR_TYPE, 2);
    })
}

/// `packed`: encode like proto3/packed writers do (one length-delimited field).
pub(super) fn attr_ints(name: &str, v: &[i64], packed: bool) -> Vec<u8> {
    msg(|w| {
        w.string(ATTR_NAME, name);
        if packed {
            let p = msg(|p| v.iter().for_each(|x| p.varint(*x as u64)));
            w.bytes(ATTR_INTS, &p);
        } else {
            v.iter().for_each(|x| w.int(ATTR_INTS, *x));
        }
        w.int(ATTR_TYPE, ATTR_TYPE_INTS);
    })
}

pub(super) fn attr_t(name: &str, t: Vec<u8>) -> Vec<u8> {
    msg(|w| {
        w.string(ATTR_NAME, name);
        w.bytes(ATTR_T, &t);
        w.int(ATTR_TYPE, 4);
    })
}

pub(super) fn node(
    op: &str,
    name: &str,
    inputs: &[&str],
    outputs: &[&str],
    attrs: &[Vec<u8>],
) -> Vec<u8> {
    msg(|w| {
        inputs.iter().for_each(|i| w.string(NODE_INPUT, i));
        outputs.iter().for_each(|o| w.string(NODE_OUTPUT, o));
        w.string(NODE_NAME, name);
        w.string(NODE_OP_TYPE, op);
        attrs.iter().for_each(|a| w.bytes(NODE_ATTRIBUTE, a));
    })
}

pub(super) fn tensor_i64(name: &str, vals: &[i64], raw: bool) -> Vec<u8> {
    msg(|w| {
        w.int(TENSOR_DIMS, vals.len() as i64);
        w.int(TENSOR_DATA_TYPE, i64::from(DT_INT64));
        if !raw {
            let p = msg(|p| vals.iter().for_each(|x| p.varint(*x as u64)));
            w.bytes(TENSOR_INT64_DATA, &p);
        }
        w.string(TENSOR_NAME, name);
        if raw {
            let r: Vec<u8> = vals.iter().flat_map(|x| x.to_le_bytes()).collect();
            w.bytes(TENSOR_RAW_DATA, &r);
        }
    })
}

pub(super) fn tensor_f32(name: &str, dims: &[i64], vals: &[f32], dtype: i32) -> Vec<u8> {
    msg(|w| {
        dims.iter().for_each(|d| w.int(TENSOR_DIMS, *d));
        w.int(TENSOR_DATA_TYPE, i64::from(dtype));
        w.string(TENSOR_NAME, name);
        let r: Vec<u8> = vals.iter().flat_map(|x| x.to_le_bytes()).collect();
        w.bytes(TENSOR_RAW_DATA, &r);
    })
}

/// `dims`: None = no shape; Some(d) with -1 for a symbolic dimension.
pub(super) fn value_info(name: &str, elem: i32, dims: Option<&[i64]>) -> Vec<u8> {
    let tt = msg(|w| {
        w.int(TT_ELEM_TYPE, i64::from(elem));
        if let Some(d) = dims {
            let shape = msg(|s| {
                for x in d {
                    let dim = msg(|dw| {
                        if *x >= 0 {
                            dw.int(DIM_VALUE, *x);
                        } else {
                            dw.string(2, "N");
                        }
                    });
                    s.bytes(SHAPE_DIM, &dim);
                }
            });
            w.bytes(TT_SHAPE, &shape);
        }
    });
    let tp = msg(|w| w.bytes(TYPE_TENSOR, &tt));
    msg(|w| {
        w.string(VI_NAME, name);
        w.bytes(VI_TYPE, &tp);
    })
}

#[derive(Default)]
pub(super) struct G {
    pub(super) nodes: Vec<Vec<u8>>,
    pub(super) inits: Vec<Vec<u8>>,
    pub(super) inputs: Vec<Vec<u8>>,
    pub(super) outputs: Vec<Vec<u8>>,
    pub(super) value_info: Vec<Vec<u8>>,
    /// raw extra graph fields (unknown fields)
    pub(super) extra: Vec<u8>,
}

pub(super) fn model(opset: i64, g: &G, extra_model: &[u8]) -> Vec<u8> {
    let graph = msg(|w| {
        g.nodes.iter().for_each(|n| w.bytes(GRAPH_NODE, n));
        w.string(2, "test");
        g.inits.iter().for_each(|t| w.bytes(GRAPH_INITIALIZER, t));
        g.inputs.iter().for_each(|v| w.bytes(GRAPH_INPUT, v));
        g.outputs.iter().for_each(|v| w.bytes(GRAPH_OUTPUT, v));
        g.value_info
            .iter()
            .for_each(|v| w.bytes(GRAPH_VALUE_INFO, v));
        w.raw(&g.extra);
    });
    let opset_id = msg(|w| {
        w.string(OPSET_DOMAIN, "");
        w.int(OPSET_VERSION, opset);
    });
    msg(|w| {
        w.int(1, 8);
        w.string(2, "bop-test");
        w.bytes(MODEL_GRAPH, &graph);
        w.bytes(MODEL_OPSET_IMPORT, &opset_id);
        w.raw(extra_model);
    })
}

/// Single-node model `x -> op -> y` with f32 input/output of `dims`.
pub(super) fn single(opset: i64, op_node: Vec<u8>, outputs: &[&str], dims: &[i64]) -> Vec<u8> {
    let g = G {
        nodes: vec![op_node],
        inputs: vec![value_info("x", DT_FLOAT, Some(dims))],
        outputs: outputs
            .iter()
            .map(|o| value_info(o, DT_FLOAT, None))
            .collect(),
        ..G::default()
    };
    model(opset, &g, &[])
}

/// (op_type, inputs, outputs, attributes as (name, f, ints))
pub(super) type ViewNode = (
    String,
    Vec<String>,
    Vec<String>,
    Vec<(String, Option<f32>, Vec<i64>)>,
);

/// Decoded view of a rewritten model for assertions.
pub(super) struct View {
    pub(super) nodes: Vec<ViewNode>,
    /// name -> (data_type, dims, raw)
    pub(super) inits: HashMap<String, (i32, Vec<i64>, Vec<u8>)>,
    pub(super) metadata: Vec<(String, String)>,
}

pub(super) fn view(bytes: &[u8]) -> View {
    let m = parse_fields(bytes).unwrap();
    let g = parse_fields(get_len(&m, MODEL_GRAPH).unwrap()).unwrap();
    let mut v = View {
        nodes: Vec::new(),
        inits: HashMap::new(),
        metadata: Vec::new(),
    };
    for f in &g {
        let b = len_of(f).unwrap_or_default();
        match f.num {
            GRAPH_NODE => {
                let n = Node::parse(b).unwrap();
                v.nodes.push((
                    n.op_type.to_string(),
                    n.inputs.iter().map(|s| s.to_string()).collect(),
                    n.outputs.iter().map(|s| s.to_string()).collect(),
                    n.attrs
                        .iter()
                        .map(|a| (a.name.to_string(), a.f, a.ints()))
                        .collect(),
                ));
            }
            GRAPH_INITIALIZER => {
                let t = Tensor::parse(b).unwrap();
                let raw = t.raw_data.unwrap_or_default().to_vec();
                v.inits
                    .insert(t.name.to_string(), (t.data_type, t.dims(), raw));
            }
            _ => {}
        }
    }
    for b in m.iter().filter(|f| f.num == MODEL_METADATA_PROPS) {
        let e = parse_fields(len_of(b).unwrap()).unwrap();
        v.metadata.push((
            get_str(&e, ENTRY_KEY).unwrap().into(),
            get_str(&e, ENTRY_VALUE).unwrap().into(),
        ));
    }
    v
}

impl View {
    pub(super) fn ops(&self) -> Vec<&str> {
        self.nodes.iter().map(|n| n.0.as_str()).collect()
    }
    pub(super) fn f32_of(&self, name: &str) -> f32 {
        let (dt, dims, raw) = &self.inits[name];
        assert_eq!((*dt, dims.len()), (DT_FLOAT, 0), "{name}: f32 scalar");
        f32::from_le_bytes(raw[..].try_into().unwrap())
    }
    pub(super) fn i64s_of(&self, name: &str) -> Vec<i64> {
        let (dt, dims, raw) = &self.inits[name];
        assert_eq!(*dt, DT_INT64, "{name}");
        assert_eq!(dims, &[raw.len() as i64 / 8], "{name}: 1-D");
        raw.as_chunks::<8>()
            .0
            .iter()
            .map(|c| i64::from_le_bytes(*c))
            .collect()
    }
}

pub(super) fn rewritten(bytes: &[u8], opts: &RewriteOptions) -> Rewrite {
    match rewrite(bytes, opts).unwrap() {
        Outcome::Rewritten(r) => r,
        other => panic!("expected a rewrite, got {other:?}"),
    }
}

/// Every rewrite path over a corpus of synthetic models, as (label, model, options).
pub(super) fn golden_corpus() -> Vec<(&'static str, Vec<u8>, RewriteOptions)> {
    let all = RewriteOptions::ALL;
    let coreml = RewriteOptions::COREML;
    let mut c = Vec::new();
    let hs = node(
        "HardSigmoid",
        "/act/HardSigmoid",
        &["x"],
        &["y"],
        &[attr_f("alpha", 1.0 / 6.0), attr_f("beta", 0.5)],
    );
    c.push(("hs12", single(12, hs, &["y"], &[1, 4]), coreml));
    let hs = node("HardSigmoid", "", &["x"], &["y"], &[]);
    c.push(("hs10", single(10, hs, &["y"], &[2]), coreml));
    let hs = node("HardSigmoid", "", &["x"], &["y"], &[]);
    c.push(("hs6", single(6, hs, &["y"], &[2]), all));
    let hs = node("HardSwish", "/act/HardSwish", &["x"], &["y"], &[]);
    c.push(("hsw17", single(17, hs, &["y"], &[1, 8]), coreml));
    for (label, dt) in [("hsw-f16", DT_FLOAT16), ("hsw-f64", DT_DOUBLE)] {
        let g = G {
            nodes: vec![node("HardSwish", "hs", &["x"], &["y"], &[])],
            inputs: vec![value_info("x", dt, Some(&[4]))],
            outputs: vec![value_info("y", dt, None)],
            ..G::default()
        };
        c.push((label, model(14, &g, &[]), coreml));
    }
    let g = G {
        nodes: vec![
            node(
                "Cast",
                "c",
                &["x"],
                &["h"],
                &[attr_i("to", i64::from(DT_FLOAT16))],
            ),
            node("HardSigmoid", "hs", &["h"], &["y"], &[]),
        ],
        inputs: vec![value_info("x", DT_FLOAT, Some(&[4]))],
        outputs: vec![value_info("y", DT_FLOAT16, None)],
        ..G::default()
    };
    c.push(("mixed", model(13, &g, &[]), coreml));
    // f16 weights make the untyped input's element type unknown.
    let g = G {
        nodes: vec![node("HardSigmoid", "hs", &["t"], &["y"], &[])],
        inits: vec![tensor_f32("w", &[1], &[1.0], DT_FLOAT16)],
        ..G::default()
    };
    c.push(("f16-weights", model(13, &g, &[]), coreml));
    let sp = node(
        "Split",
        "split",
        &["x"],
        &["a", "b", "c"],
        &[attr_i("axis", 4), attr_ints("split", &[2, 2, 4], false)],
    );
    c.push((
        "split12",
        single(12, sp, &["a", "b", "c"], &[1, 3, 80, 80, 8]),
        coreml,
    ));
    let sp = node(
        "Split",
        "split",
        &["x"],
        &["a", "b"],
        &[attr_i("axis", -1), attr_ints("split", &[3, 5], true)],
    );
    c.push(("split12-packed", single(12, sp, &["a", "b"], &[8]), coreml));
    let sp = node(
        "Split",
        "split",
        &["x"],
        &["a", "b"],
        &[attr_i("axis", 1), attr_ints("split", &[1, 3], false)],
    );
    c.push(("split9", single(9, sp, &["a", "b"], &[2, 4]), coreml));
    let sp = node("Split", "s", &["x"], &["a", "b"], &[attr_i("axis", 1)]);
    c.push(("split11-eq", single(11, sp, &["a", "b"], &[1, 6]), coreml));
    let sp = node("Split", "s", &["x"], &["a", "b"], &[attr_i("axis", 1)]);
    c.push(("split11-sym", single(11, sp, &["a", "b"], &[1, -1]), coreml));
    for (label, raw, constant) in [
        ("split13-raw", true, false),
        ("split13-typed", false, false),
        ("split13-const", true, true),
    ] {
        let sizes = tensor_i64("sizes", &[2, 6], raw);
        let mut g = G {
            nodes: vec![node(
                "Split",
                "s",
                &["x", "sizes"],
                &["a", "b"],
                &[attr_i("axis", 0)],
            )],
            inputs: vec![value_info("x", DT_FLOAT, Some(&[8]))],
            outputs: vec![
                value_info("a", DT_FLOAT, None),
                value_info("b", DT_FLOAT, None),
            ],
            ..G::default()
        };
        if constant {
            g.nodes.insert(
                0,
                node("Constant", "k", &[], &["sizes"], &[attr_t("value", sizes)]),
            );
        } else {
            g.inits.push(sizes);
        }
        c.push((label, model(13, &g, &[]), all));
    }
    // int32 sizes, raw and typed (packed and not).
    let i32_raw = msg(|w| {
        w.int(TENSOR_DIMS, 2);
        w.int(TENSOR_DATA_TYPE, i64::from(DT_INT32));
        w.string(TENSOR_NAME, "sizes");
        w.bytes(TENSOR_RAW_DATA, &[3, 0, 0, 0, 5, 0, 0, 0]);
    });
    let i32_typed = msg(|w| {
        w.int(TENSOR_DIMS, 2);
        w.int(TENSOR_DATA_TYPE, i64::from(DT_INT32));
        w.int(TENSOR_INT32_DATA, 3);
        w.int(TENSOR_INT32_DATA, 5);
        w.string(TENSOR_NAME, "sizes");
    });
    for (label, t) in [("split13-i32raw", i32_raw), ("split13-i32typed", i32_typed)] {
        let g = G {
            nodes: vec![node("Split", "s", &["x", "sizes"], &["a", "b"], &[])],
            inits: vec![t],
            inputs: vec![value_info("x", DT_FLOAT, Some(&[8]))],
            ..G::default()
        };
        c.push((label, model(13, &g, &[]), all));
    }
    let g = G {
        nodes: vec![
            node("Split", "s", &["x", "sizes"], &["a", "b"], &[]),
            node("Identity", "i", &["sizes"], &["c"], &[]),
        ],
        inits: vec![tensor_i64("sizes", &[2, 6], true)],
        inputs: vec![value_info("x", DT_FLOAT, Some(&[8]))],
        ..G::default()
    };
    c.push(("split13-kept", model(13, &g, &[]), all));
    let g = G {
        nodes: vec![
            node(
                "Constant",
                "k",
                &[],
                &["sizes"],
                &[attr_ints("value_ints", &[4, 4], false)],
            ),
            node("Split", "s", &["x", "sizes"], &["a", "b"], &[]),
        ],
        inputs: vec![value_info("x", DT_FLOAT, Some(&[8]))],
        ..G::default()
    };
    c.push(("split13-value-ints", model(13, &g, &[]), all));
    let sp = node(
        "Split",
        "s",
        &["x"],
        &["a", "b", "c"],
        &[attr_i("axis", -1), attr_i("num_outputs", 3)],
    );
    c.push(("split18", single(18, sp, &["a", "b", "c"], &[2, 10]), all));
    let g = G {
        nodes: vec![
            node("Relu", "r", &["x"], &["t"], &[]),
            node(
                "Split",
                "s",
                &["t"],
                &["a", "b"],
                &[attr_i("axis", 0), attr_i("num_outputs", 2)],
            ),
        ],
        inputs: vec![value_info("x", DT_FLOAT, None)],
        value_info: vec![value_info("t", DT_FLOAT, Some(&[6, 2]))],
        ..G::default()
    };
    c.push(("split18-vi", model(18, &g, &[]), all));
    for (label, axis, dims) in [
        ("flat-none", None, &[-1i64, 3, 4][..]),
        ("flat0", Some(0), &[-1, 3][..]),
        ("flat2", Some(2), &[2, 3, 4, 5][..]),
        ("flat-1", Some(-1), &[2, 3, 4][..]),
        ("flat2-sym", Some(2), &[-1, 3, 4][..]),
    ] {
        let attrs: Vec<_> = axis.map(|a| attr_i("axis", a)).into_iter().collect();
        let f = node("Flatten", "f", &["x"], &["y"], &attrs);
        c.push((label, single(13, f, &["y"], dims), all));
    }
    let mut unknown = Writer::default();
    unknown.string(999, "keep me");
    unknown.int(1000, 42);
    let mut relu = node("Relu", "relu", &["y"], &["z"], &[]);
    relu.extend_from_slice(&unknown.buf);
    let g = G {
        nodes: vec![node("HardSigmoid", "hs", &["x"], &["y"], &[]), relu],
        inits: vec![tensor_f32("w", &[2], &[1.0, 2.0], DT_FLOAT)],
        inputs: vec![value_info("x", DT_FLOAT, Some(&[1, 2]))],
        outputs: vec![value_info("z", DT_FLOAT, None)],
        extra: unknown.buf.clone(),
        ..G::default()
    };
    c.push(("unknown-fields", model(13, &g, &unknown.buf), coreml));
    let g = G {
        nodes: vec![
            node(
                "Relu",
                "bop_rewrite/const_0",
                &["x"],
                &["bop_rewrite/const_0"],
                &[],
            ),
            node("HardSigmoid", "hs", &["bop_rewrite/const_0"], &["y"], &[]),
            msg(|w| {
                w.string(NODE_INPUT, "y");
                w.string(NODE_OUTPUT, "z");
                w.string(NODE_OP_TYPE, "HardSigmoid");
                w.string(NODE_DOMAIN, "com.example");
            }),
        ],
        inputs: vec![value_info("x", DT_FLOAT, Some(&[2]))],
        ..G::default()
    };
    c.push(("names", model(13, &g, &[]), all));
    let ext = msg(|w| {
        w.int(TENSOR_DIMS, 2);
        w.int(TENSOR_DATA_TYPE, i64::from(DT_FLOAT));
        w.string(TENSOR_NAME, "w");
        w.int(TENSOR_DATA_LOCATION, 1);
    });
    let g = G {
        nodes: vec![node("HardSigmoid", "hs", &["x"], &["y"], &[])],
        inits: vec![ext],
        inputs: vec![value_info("x", DT_FLOAT, Some(&[2]))],
        ..G::default()
    };
    c.push(("external", model(13, &g, &[]), all));
    c
}

pub(super) fn outcome_digest(o: &Outcome) -> String {
    match o {
        Outcome::Unchanged { skipped } => format!("U {}", skipped.join(" | ")),
        Outcome::ExternalData => "E".to_string(),
        Outcome::Rewritten(r) => format!(
            "R {} {} {}",
            &sha256_hex(&r.bytes)[..16],
            summary(&r.counts),
            r.skipped.join(" | ")
        ),
    }
}

/// A HardSwish model with a 100 MB raw f32 weight and a 1M-entry unpacked `float_data` weight
/// (one encoded field per value: the worst case for a per-field parse), or without weights.
pub(super) fn big_model(weights: bool) -> Vec<u8> {
    let raw = msg(|w| {
        w.int(TENSOR_DIMS, 25 << 20);
        w.int(TENSOR_DATA_TYPE, i64::from(DT_FLOAT));
        w.string(TENSOR_NAME, "big_raw");
        w.bytes(TENSOR_RAW_DATA, &vec![0x3f; 100 << 20]);
    });
    let unpacked = msg(|w| {
        w.int(TENSOR_DIMS, 1 << 20);
        w.int(TENSOR_DATA_TYPE, i64::from(DT_FLOAT));
        w.string(TENSOR_NAME, "big_unpacked");
        for i in 0..1u32 << 20 {
            #[allow(clippy::cast_precision_loss, reason = "test values")]
            w.float(4, i as f32);
        }
    });
    let g = G {
        nodes: vec![node("HardSwish", "hs", &["x"], &["y"], &[])],
        inits: if weights {
            vec![raw, unpacked]
        } else {
            Vec::new()
        },
        inputs: vec![value_info("x", DT_FLOAT, Some(&[1, 4]))],
        outputs: vec![value_info("y", DT_FLOAT, None)],
        ..G::default()
    };
    model(13, &g, &[])
}

/// Fields `parse_fields` stored while rewriting `m`.
pub(super) fn stored_fields(m: &[u8]) -> (usize, Rewrite) {
    FIELDS_STORED.with(|c| c.set(0));
    let r = rewritten(m, &RewriteOptions::COREML);
    (FIELDS_STORED.with(std::cell::Cell::get), r)
}
