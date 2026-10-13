//! Graph rewrites that let the ONNX Runtime CoreML execution provider take a whole model.
//!
//! The CoreML EP (ONNX Runtime 1.24) has no `HardSigmoid`/`HardSwish` builder and accepts `Split`
//! only from opset 13 (`split` as an input) and `Flatten` not at all in MLProgram mode. YOLOv5
//! exports use Hardswish in every layer, so the graph falls apart into ~50 CoreML partitions with
//! CPU hops in between and runs slower than the plain CPU. Replacing those nodes by equivalent
//! elementwise ops CoreML supports makes it one partition:
//!
//! - `HardSigmoid(x; alpha, beta)` -> `Clip(Add(Mul(x, alpha), beta), 0, 1)`
//! - `HardSwish(x)` -> `Mul(x, Clip(Add(Mul(x, 1/6), 0.5), 0, 1))`
//! - `Split(x; axis, split)` -> one `Slice` per output (starts/ends/axes as int64 initializers)
//! - `Flatten(x; axis)` -> `Reshape(x, [0 | 1 | d0*..*d(axis-1), -1])` (opt-in: not part of
//!   [`RewriteOptions::COREML`], since on yolo26n it raised the partition count from 5 to 6
//!   without a speed-up; its other unsupported ops stay anyway)
//!
//! Clip takes min/max as inputs from opset 11 and as attributes before; Slice takes
//! starts/ends/axes as inputs from opset 10 and as attributes before.
//!
//! The model is handled at the protobuf wire-format level: every field the rewrite does not touch
//! (weights, doc strings, metadata, unknown fields, subgraphs) is copied byte for byte, only
//! `ModelProto.graph` and the replaced `GraphProto.node`s are re-encoded and new initializers are
//! appended. A model without anything to rewrite is reported as [`Outcome::Unchanged`]; models
//! with external data are left alone ([`Outcome::ExternalData`]), since a copy elsewhere would
//! lose the relative data paths. [`rewrite_cached`] writes the result next to other caches.
//!
//! Layout: `proto` (protobuf wire format), `onnx` (the ONNX messages the rewrite reads), `emit`
//! (graph analysis, replacement nodes, re-serialization); this file holds the options, the
//! `rewrite` entry point and the cached copies.

mod emit;
mod onnx;
mod proto;
#[cfg(test)]
mod test_models;

pub use emit::reencode;
pub use onnx::{DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT64};

use crate::resources::manager::sha256_hex;
use anyhow::{Context, Result, bail};
use emit::{Builder, Emit, Graph, assemble, emit_flatten, emit_hard_sigmoid, emit_split};
use onnx::{MODEL_GRAPH, onnx_opset};
use proto::{len_of, parse_fields};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};

/// Bumped whenever the rewrite output changes, so cached copies are regenerated.
pub const REWRITER_VERSION: u32 = 1;

// ---- options and results -----------------------------------------------------------------------

/// Which `Split` nodes to replace by `Slice`s.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitRewrite {
    Never,
    /// Only below opset 13 (sizes as the `split` attribute), which the CoreML EP rejects.
    BelowOpset13,
    Always,
}

/// Which rewrites to apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RewriteOptions {
    pub hard_sigmoid: bool,
    pub hard_swish: bool,
    pub split: SplitRewrite,
    pub flatten: bool,
}

impl RewriteOptions {
    /// What the ONNX Runtime 1.24 CoreML EP (MLProgram) cannot take but supports once rewritten.
    /// Measured on YOLOv5 (IPcam-*, delivery, package, ipcam-bird): one CoreML partition instead
    /// of 2-53, outputs bit-identical on the CPU EP.
    pub const COREML: Self = Self {
        hard_sigmoid: true,
        hard_swish: true,
        split: SplitRewrite::BelowOpset13,
        flatten: false,
    };
    /// Every rewrite (tests).
    pub const ALL: Self = Self {
        hard_sigmoid: true,
        hard_swish: true,
        split: SplitRewrite::Always,
        flatten: true,
    };
}

/// A rewritten model.
#[derive(Debug, Clone)]
pub struct Rewrite {
    pub bytes: Vec<u8>,
    /// Rewritten nodes per op type.
    pub counts: BTreeMap<String, usize>,
    /// Nodes that matched but could not be rewritten, with the reason.
    pub skipped: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum Outcome {
    /// Nothing to rewrite (`skipped`: candidates that could not be rewritten).
    Unchanged {
        skipped: Vec<String>,
    },
    /// The model keeps its weights in external files; left alone.
    ExternalData,
    Rewritten(Rewrite),
}

/// "HardSigmoid x51, Split x3".
pub fn summary(counts: &BTreeMap<String, usize>) -> String {
    counts
        .iter()
        .map(|(k, v)| format!("{k} x{v}"))
        .collect::<Vec<_>>()
        .join(", ")
}

// ---- rewrite -----------------------------------------------------------------------------------

/// Rewrite `model` (serialized ModelProto) as `opts` asks.
pub fn rewrite(model: &[u8], opts: &RewriteOptions) -> Result<Outcome> {
    let mfields = parse_fields(model).context("parsing ModelProto")?;
    let Some(gi) = mfields.iter().rposition(|f| f.num == MODEL_GRAPH) else {
        bail!("model has no graph");
    };
    let gbytes = mfields
        .get(gi)
        .and_then(len_of)
        .context("graph field is not length-delimited")?;
    let opset = onnx_opset(&mfields)?;
    let mut g = Graph::parse(gbytes).context("parsing GraphProto")?;
    if g.external_data {
        return Ok(Outcome::ExternalData);
    }
    let mut skipped = Vec::new();
    let Some(opset) = opset else {
        return Ok(Outcome::Unchanged { skipped });
    };

    let mut names = std::mem::take(&mut g.names);
    let mut b = Builder {
        names: &mut names,
        inits: Vec::new(),
        consts: HashMap::new(),
    };
    // field index of a replaced node -> replacement nodes
    let mut replaced: HashMap<usize, Vec<Vec<u8>>> = HashMap::new();
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut maybe_dead: Vec<&str> = Vec::new();
    for (fi, n) in &g.nodes {
        if !n.is_onnx() {
            continue;
        }
        let wanted = match n.op_type {
            "HardSigmoid" => opts.hard_sigmoid,
            "HardSwish" => opts.hard_swish,
            "Split" => match opts.split {
                SplitRewrite::Never => false,
                SplitRewrite::BelowOpset13 => opset < 13,
                SplitRewrite::Always => true,
            },
            "Flatten" => opts.flatten,
            _ => false,
        };
        if !wanted {
            continue;
        }
        let base = if n.name.is_empty() {
            format!("bop_rewrite/{}", n.op_type)
        } else {
            n.name.to_string()
        };
        let mut e = Emit {
            b: &mut b,
            base,
            nodes: Vec::new(),
        };
        let r = match n.op_type {
            "HardSigmoid" | "HardSwish" => {
                let swish = n.op_type == "HardSwish";
                match (n.input(0), n.outputs.first()) {
                    (Some(x), Some(y)) if opset >= 7 => match g.elem_type(x) {
                        Some(dt) => {
                            let (alpha, beta) = if swish {
                                (1.0 / 6.0, 0.5)
                            } else {
                                (
                                    n.attr("alpha").and_then(|a| a.f).unwrap_or(0.2),
                                    n.attr("beta").and_then(|a| a.f).unwrap_or(0.5),
                                )
                            };
                            emit_hard_sigmoid(&mut e, opset, dt, x, (alpha, beta), y, swish)
                        }
                        None => Err(format!("element type of '{x}' unknown")),
                    },
                    _ if opset < 7 => Err(format!("opset {opset} < 7")),
                    _ => Err("missing input or output".to_string()),
                }
            }
            "Split" => emit_split(&mut e, &g, n, opset),
            "Flatten" if opset >= 5 => emit_flatten(&mut e, &g, n),
            "Flatten" => Err(format!("opset {opset} < 5")),
            _ => unreachable!("filtered above"),
        };
        match r {
            Ok(()) => {
                *counts.entry(n.op_type.to_string()).or_default() += 1;
                replaced.insert(*fi, e.nodes);
                if n.op_type == "Split"
                    && opset >= 13
                    && let Some(s) = n.input(1)
                {
                    maybe_dead.push(s);
                }
            }
            Err(why) => {
                // Names minted for a failed attempt stay reserved; harmless.
                let label = if n.name.is_empty() {
                    "<unnamed>"
                } else {
                    n.name
                };
                skipped.push(format!("{} '{label}': {why}", n.op_type));
            }
        }
    }
    if replaced.is_empty() {
        return Ok(Outcome::Unchanged { skipped });
    }
    let new_inits = std::mem::take(&mut b.inits);

    // Split sizes nobody reads any more: drop their initializer / Constant node (ONNX Runtime
    // warns about unused initializers). Not attempted when subgraphs could reference them.
    let mut dropped: HashSet<usize> = HashSet::new();
    if !g.has_subgraphs {
        let used: HashSet<&str> = g
            .nodes
            .iter()
            .filter(|(fi, _)| !replaced.contains_key(fi))
            .flat_map(|(_, n)| n.inputs.iter().copied())
            .chain(g.outputs.iter().copied())
            .chain(g.inputs.iter().copied())
            .collect();
        for name in maybe_dead {
            if used.contains(name) {
                continue;
            }
            if let Some((fi, _)) = g.inits.get(name) {
                dropped.insert(*fi);
            } else if let Some((fi, _)) = g.nodes.iter().find(|(_, n)| {
                n.op_type == "Constant" && n.outputs.len() == 1 && n.outputs.first() == Some(&name)
            }) {
                dropped.insert(*fi);
            }
        }
    }

    let marker = format!("v{REWRITER_VERSION}: {}", summary(&counts));
    let bytes = assemble(
        &mfields,
        gi,
        &g.fields,
        &replaced,
        &dropped,
        &new_inits,
        Some(&marker),
    );
    Ok(Outcome::Rewritten(Rewrite {
        bytes,
        counts,
        skipped,
    }))
}

// ---- cached copies -----------------------------------------------------------------------------

/// Result of [`rewrite_cached`].
#[derive(Debug, Clone)]
pub enum Cached {
    /// Use the original model (`skipped`: candidates that could not be rewritten).
    Original { skipped: Vec<String> },
    /// The model has external data; use the original.
    ExternalData,
    /// Load `path` instead. `created` is false when an earlier copy was reused (then `counts`
    /// are recomputed all the same, so logs stay informative).
    Rewritten {
        path: PathBuf,
        counts: BTreeMap<String, usize>,
        skipped: Vec<String>,
        created: bool,
    },
}

/// Cache file name for a rewrite of `src` with contents hash `sha_hex`.
pub fn cache_file_name(src: &Path, sha_hex: &str) -> String {
    let stem = src
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "model".into());
    let short = &sha_hex[..sha_hex.len().min(16)];
    format!("{stem}-{short}-v{REWRITER_VERSION}.onnx")
}

/// Largest model file [`rewrite_cached`] reads (512 MiB). The rewrite holds the file, its
/// rewritten copy and the parsed graph in memory at once (about 2x the file size plus a small
/// per-node overhead), so a bigger model loads unrewritten instead; models that large normally
/// keep their weights in external data, which is never rewritten anyway (BE-06).
pub const MAX_REWRITE_BYTES: u64 = 512 << 20;

/// The contents of `path`, or `None` when it is larger than `max` bytes. The size is checked on
/// the open handle and the read is capped, so a file that grows meanwhile is not read whole.
fn read_capped(path: &Path, max: u64) -> std::io::Result<Option<Vec<u8>>> {
    use std::io::Read;
    let file = std::fs::File::open(path)?;
    let len = file.metadata()?.len();
    if len > max {
        return Ok(None);
    }
    let mut bytes = Vec::with_capacity(usize::try_from(len).unwrap_or(0));
    file.take(max.saturating_add(1)).read_to_end(&mut bytes)?;
    Ok((bytes.len() as u64 <= max).then_some(bytes))
}

/// Whether the file at `path` holds exactly `expected`, compared in chunks (the existing copy
/// is never read into memory whole; a size mismatch is decided from the metadata).
fn file_equals(path: &Path, expected: &[u8]) -> std::io::Result<bool> {
    use std::io::Read;
    let file = std::fs::File::open(path)?;
    if file.metadata()?.len() != expected.len() as u64 {
        return Ok(false);
    }
    let mut reader = std::io::BufReader::new(file);
    let mut buf = vec![0u8; 1 << 16];
    let mut rest = expected;
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            return Ok(rest.is_empty());
        }
        let (Some(got), Some(want)) = (buf.get(..n), rest.get(..n)) else {
            return Ok(false); // the file grew past the expected length
        };
        if got != want {
            return Ok(false);
        }
        rest = rest.get(n..).unwrap_or_default();
    }
}

/// Rewrite `src` into `dir/<stem>-<sha256 16 hex>-v<version>.onnx` (written to a temporary file
/// and renamed into place; reused when it exists). `src` itself is never modified. A model
/// larger than [`MAX_REWRITE_BYTES`] is not read and reported as [`Cached::Original`].
pub fn rewrite_cached(src: &Path, dir: &Path, opts: &RewriteOptions) -> Result<Cached> {
    rewrite_cached_capped(src, dir, opts, MAX_REWRITE_BYTES)
}

fn rewrite_cached_capped(
    src: &Path,
    dir: &Path,
    opts: &RewriteOptions,
    max_bytes: u64,
) -> Result<Cached> {
    let Some(bytes) =
        read_capped(src, max_bytes).with_context(|| format!("reading {}", src.display()))?
    else {
        let size = std::fs::metadata(src).map(|m| m.len()).unwrap_or(0);
        tracing::info!(
            "model {}: {size} bytes is over the CoreML rewrite limit of {max_bytes} bytes \
             (MAX_REWRITE_BYTES); loading it without the graph rewrite",
            src.display()
        );
        return Ok(Cached::Original {
            skipped: Vec::new(),
        });
    };
    let r = match rewrite(&bytes, opts).with_context(|| format!("parsing {}", src.display()))? {
        Outcome::Unchanged { skipped } => return Ok(Cached::Original { skipped }),
        Outcome::ExternalData => return Ok(Cached::ExternalData),
        Outcome::Rewritten(r) => r,
    };
    let path = dir.join(cache_file_name(src, &sha256_hex(&bytes)));
    drop(bytes);
    // A cache hit is trusted only when its bytes are exactly the rewrite just computed (the
    // rewrite is deterministic): a file somebody else put at this predictable path, or a
    // truncated / stale one, is replaced (BE-04).
    match file_equals(&path, &r.bytes) {
        Ok(true) => {
            return Ok(Cached::Rewritten {
                path,
                counts: r.counts,
                skipped: r.skipped,
                created: false,
            });
        }
        Ok(false) => tracing::warn!(
            "cached CoreML rewrite {} does not match the rewrite of {}; replacing it",
            path.display(),
            src.display()
        ),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => tracing::debug!(
            "cached rewrite {} unreadable ({e}); rewriting",
            path.display()
        ),
    }
    std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let tmp = dir.join(format!(
        ".{}.tmp-{}-{}",
        path.file_name().unwrap_or_default().to_string_lossy(),
        std::process::id(),
        SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    let write = || -> Result<()> {
        std::fs::write(&tmp, &r.bytes).with_context(|| format!("writing {}", tmp.display()))?;
        std::fs::rename(&tmp, &path)
            .with_context(|| format!("renaming {} to {}", tmp.display(), path.display()))
    };
    if let Err(e) = write() {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    Ok(Cached::Rewritten {
        path,
        counts: r.counts,
        skipped: r.skipped,
        created: true,
    })
}

// ---- tests -------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::onnx::*;
    use super::proto::*;
    use super::test_models::*;
    use super::*;

    #[test]
    fn summary_lists_counts() {
        let mut c = BTreeMap::new();
        c.insert("Split".to_string(), 3);
        c.insert("HardSigmoid".to_string(), 51);
        assert_eq!(summary(&c), "HardSigmoid x51, Split x3");
    }

    #[test]
    fn unknown_fields_and_untouched_nodes_are_preserved() {
        let mut unknown = Writer::default();
        unknown.string(999, "keep me");
        unknown.int(1000, 42);
        // An untouched node with an unknown field of its own.
        let mut relu = node("Relu", "relu", &["y"], &["z"], &[]);
        relu.extend_from_slice(&unknown.buf);
        let hs = node("HardSigmoid", "hs", &["x"], &["y"], &[]);
        let weights = tensor_f32("w", &[2], &[1.0, 2.0], DT_FLOAT);
        let g = G {
            nodes: vec![hs, relu.clone()],
            inits: vec![weights.clone()],
            inputs: vec![value_info("x", DT_FLOAT, Some(&[1, 2]))],
            outputs: vec![value_info("z", DT_FLOAT, None)],
            extra: unknown.buf.clone(),
            ..G::default()
        };
        let m = model(13, &g, &unknown.buf);
        let r = rewritten(&m, &RewriteOptions::COREML);
        let contains = |hay: &[u8], needle: &[u8]| hay.windows(needle.len()).any(|w| w == needle);
        let out = parse_fields(&r.bytes).unwrap();
        // Model-level unknown fields, verbatim.
        assert!(
            out.iter()
                .any(|f| f.num == 999 && f.value == Value::Len(b"keep me"))
        );
        assert!(
            out.iter()
                .any(|f| f.num == 1000 && f.value == Value::Varint(42))
        );
        let gb = get_len(&out, MODEL_GRAPH).unwrap();
        let gf = parse_fields(gb).unwrap();
        assert!(
            gf.iter()
                .any(|f| f.num == 999 && f.value == Value::Len(b"keep me"))
        );
        assert!(gf.iter().any(|f| f.num == 1000));
        // The untouched node and the weights, byte for byte.
        assert!(contains(gb, &relu));
        assert!(contains(gb, &weights));
        assert_eq!(get_str(&gf, 2).unwrap(), "test");
        assert_eq!(get_int(&out, 1), Some(8));
        let v = view(&r.bytes);
        assert_eq!(v.ops(), ["Mul", "Add", "Clip", "Relu"]);
        assert_eq!(
            v.metadata,
            [(
                "blue_onyx_prism.rewrite".to_string(),
                format!("v{REWRITER_VERSION}: HardSigmoid x1")
            )]
        );
    }

    #[test]
    fn hard_sigmoid_opset_11_plus_uses_clip_inputs() {
        let hs = node(
            "HardSigmoid",
            "/act/HardSigmoid",
            &["x"],
            &["y"],
            &[attr_f("alpha", 1.0 / 6.0), attr_f("beta", 0.5)],
        );
        let m = single(12, hs, &["y"], &[1, 4]);
        let r = rewritten(&m, &RewriteOptions::COREML);
        assert_eq!(r.counts.get("HardSigmoid"), Some(&1));
        let v = view(&r.bytes);
        assert_eq!(v.ops(), ["Mul", "Add", "Clip"]);
        let (mul, add, clip) = (&v.nodes[0], &v.nodes[1], &v.nodes[2]);
        assert_eq!(mul.1[0], "x");
        assert_eq!(v.f32_of(&mul.1[1]), 1.0f32 / 6.0);
        assert_eq!(add.1[0], mul.2[0]);
        assert_eq!(v.f32_of(&add.1[1]), 0.5);
        assert_eq!(clip.1[0], add.2[0]);
        assert_eq!(v.f32_of(&clip.1[1]), 0.0);
        assert_eq!(v.f32_of(&clip.1[2]), 1.0);
        assert_eq!(clip.2, ["y"]);
        assert!(clip.3.is_empty());
        // Shared constants: 1/6, 0.5, 0 and 1.
        assert_eq!(v.inits.len(), 4);
        // Fresh names, never reusing graph names.
        assert!(mul.2[0].starts_with("y/bop_mul"));
    }

    #[test]
    fn hard_sigmoid_defaults_and_old_clip() {
        // alpha/beta default to 0.2/0.5; before opset 11 Clip takes min/max attributes.
        let hs = node("HardSigmoid", "", &["x"], &["y"], &[]);
        let m = single(10, hs, &["y"], &[2]);
        let v = view(&rewritten(&m, &RewriteOptions::COREML).bytes);
        assert_eq!(v.ops(), ["Mul", "Add", "Clip"]);
        assert_eq!(v.f32_of(&v.nodes[0].1[1]), 0.2);
        assert_eq!(v.f32_of(&v.nodes[1].1[1]), 0.5);
        let clip = &v.nodes[2];
        assert_eq!(clip.1.len(), 1);
        assert_eq!(
            clip.3,
            [
                ("min".to_string(), Some(0.0), vec![]),
                ("max".to_string(), Some(1.0), vec![])
            ]
        );
        assert_eq!(v.inits.len(), 2);
        // Before opset 7 (no broadcasting Mul/Add) nothing is rewritten.
        let hs = node("HardSigmoid", "", &["x"], &["y"], &[]);
        match rewrite(&single(6, hs, &["y"], &[2]), &RewriteOptions::ALL).unwrap() {
            Outcome::Unchanged { skipped } => {
                assert!(skipped[0].contains("opset 6"), "{skipped:?}")
            }
            o => panic!("{o:?}"),
        }
    }

    #[test]
    fn hard_swish_multiplies_by_x() {
        let hs = node("HardSwish", "/act/HardSwish", &["x"], &["y"], &[]);
        let m = single(17, hs, &["y"], &[1, 8]);
        let r = rewritten(&m, &RewriteOptions::COREML);
        assert_eq!(r.counts.get("HardSwish"), Some(&1));
        let v = view(&r.bytes);
        assert_eq!(v.ops(), ["Mul", "Add", "Clip", "Mul"]);
        assert_eq!(v.f32_of(&v.nodes[0].1[1]), 1.0f32 / 6.0);
        assert_eq!(v.f32_of(&v.nodes[1].1[1]), 0.5);
        let last = &v.nodes[3];
        assert_eq!(last.1, ["x".to_string(), v.nodes[2].2[0].clone()]);
        assert_eq!(last.2, ["y"]);
        assert_ne!(v.nodes[2].2[0], "y");
    }

    #[test]
    fn float16_and_double_scalars_follow_the_input_type() {
        for (dt, size) in [(DT_FLOAT16, 2usize), (DT_DOUBLE, 8)] {
            let g = G {
                nodes: vec![node("HardSwish", "hs", &["x"], &["y"], &[])],
                inputs: vec![value_info("x", dt, Some(&[4]))],
                outputs: vec![value_info("y", dt, None)],
                ..G::default()
            };
            let v = view(&rewritten(&model(14, &g, &[]), &RewriteOptions::COREML).bytes);
            let (t, dims, raw) = &v.inits[&v.nodes[0].1[1]];
            assert_eq!((*t, dims.len(), raw.len()), (dt, 0, size));
            if dt == DT_FLOAT16 {
                assert_eq!(raw[..], 0x3155u16.to_le_bytes());
            } else {
                assert_eq!(
                    f64::from_le_bytes(raw[..].try_into().unwrap()),
                    f64::from(1.0f32 / 6.0)
                );
            }
        }
    }

    #[test]
    fn unknown_element_type_in_mixed_precision_model_is_skipped() {
        // The HardSigmoid input has no recorded type and the model also holds f16 data: the
        // scalar type cannot be chosen safely.
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
        match rewrite(&model(13, &g, &[]), &RewriteOptions::COREML).unwrap() {
            Outcome::Unchanged { skipped } => {
                assert_eq!(skipped.len(), 1);
                assert!(
                    skipped[0].contains("element type of 'h' unknown"),
                    "{skipped:?}"
                );
            }
            o => panic!("{o:?}"),
        }
    }

    #[test]
    fn split_attribute_form_becomes_slices() {
        // YOLOv5 head: Split(axis=4, split=[2,2,4]) on a 5-D tensor, opset 12.
        let sp = node(
            "Split",
            "split",
            &["x"],
            &["a", "b", "c"],
            &[attr_i("axis", 4), attr_ints("split", &[2, 2, 4], false)],
        );
        let m = single(12, sp, &["a", "b", "c"], &[1, 3, 80, 80, 8]);
        let r = rewritten(&m, &RewriteOptions::COREML);
        assert_eq!(r.counts.get("Split"), Some(&1));
        let v = view(&r.bytes);
        assert_eq!(v.ops(), ["Slice", "Slice", "Slice"]);
        let mut ranges = Vec::new();
        for (j, out) in ["a", "b", "c"].iter().enumerate() {
            let n = &v.nodes[j];
            assert_eq!(n.1.len(), 4);
            assert_eq!(n.1[0], "x");
            assert_eq!(n.2, [out.to_string()]);
            assert_eq!(v.i64s_of(&n.1[3]), [4]);
            ranges.push((v.i64s_of(&n.1[1])[0], v.i64s_of(&n.1[2])[0]));
        }
        assert_eq!(ranges, [(0, 2), (2, 4), (4, 8)]);
        // starts 0/2/4, ends 2/4/8 and axes 4, shared: 0,2,4,8 -> 4 tensors ([4] doubles as axes).
        assert_eq!(v.inits.len(), 4);
        // Packed `split` ints parse the same.
        let sp = node(
            "Split",
            "split",
            &["x"],
            &["a", "b"],
            &[attr_i("axis", -1), attr_ints("split", &[3, 5], true)],
        );
        let v = view(&rewritten(&single(12, sp, &["a", "b"], &[8]), &RewriteOptions::COREML).bytes);
        assert_eq!(v.i64s_of(&v.nodes[1].1[1]), [3]);
        assert_eq!(v.i64s_of(&v.nodes[1].1[2]), [8]);
        assert_eq!(v.i64s_of(&v.nodes[1].1[3]), [-1]);
    }

    #[test]
    fn split_before_opset_10_uses_slice_attributes() {
        let sp = node(
            "Split",
            "split",
            &["x"],
            &["a", "b"],
            &[attr_i("axis", 1), attr_ints("split", &[1, 3], false)],
        );
        let v = view(
            &rewritten(
                &single(9, sp, &["a", "b"], &[2, 4]),
                &RewriteOptions::COREML,
            )
            .bytes,
        );
        assert_eq!(v.ops(), ["Slice", "Slice"]);
        assert!(v.inits.is_empty());
        assert_eq!(v.nodes[1].1, ["x"]);
        assert_eq!(
            v.nodes[1].3,
            [
                ("starts".to_string(), None, vec![1]),
                ("ends".to_string(), None, vec![4]),
                ("axes".to_string(), None, vec![1])
            ]
        );
    }

    #[test]
    fn equal_split_needs_the_dimension() {
        // Opset 11 without `split`: equal parts of the known dimension.
        let sp = node("Split", "s", &["x"], &["a", "b"], &[attr_i("axis", 1)]);
        let v = view(
            &rewritten(
                &single(11, sp, &["a", "b"], &[1, 6]),
                &RewriteOptions::COREML,
            )
            .bytes,
        );
        assert_eq!(v.i64s_of(&v.nodes[1].1[1]), [3]);
        assert_eq!(v.i64s_of(&v.nodes[1].1[2]), [6]);
        // Unknown (symbolic) dimension: skipped with a reason.
        let sp = node("Split", "s", &["x"], &["a", "b"], &[attr_i("axis", 1)]);
        match rewrite(
            &single(11, sp, &["a", "b"], &[1, -1]),
            &RewriteOptions::COREML,
        )
        .unwrap()
        {
            Outcome::Unchanged { skipped } => {
                assert!(skipped[0].contains("needs its shape"), "{skipped:?}")
            }
            o => panic!("{o:?}"),
        }
    }

    #[test]
    fn split_opset_13_input_from_initializer_or_constant() {
        for (raw, constant) in [(true, false), (false, false), (true, true)] {
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
            let m = model(13, &g, &[]);
            // CoreML takes opset-13 Split as is.
            assert!(matches!(
                rewrite(&m, &RewriteOptions::COREML).unwrap(),
                Outcome::Unchanged { .. }
            ));
            let v = view(&rewritten(&m, &RewriteOptions::ALL).bytes);
            // The sizes tensor/Constant is dropped once no node reads it.
            assert_eq!(v.ops(), ["Slice", "Slice"], "raw {raw} constant {constant}");
            assert!(!v.inits.contains_key("sizes"));
            assert_eq!(v.i64s_of(&v.nodes[1].1[1]), [2]);
            assert_eq!(v.i64s_of(&v.nodes[1].1[2]), [8]);
        }
        // Still read by another node: kept.
        let g = G {
            nodes: vec![
                node("Split", "s", &["x", "sizes"], &["a", "b"], &[]),
                node("Identity", "i", &["sizes"], &["c"], &[]),
            ],
            inits: vec![tensor_i64("sizes", &[2, 6], true)],
            inputs: vec![value_info("x", DT_FLOAT, Some(&[8]))],
            ..G::default()
        };
        let v = view(&rewritten(&model(13, &g, &[]), &RewriteOptions::ALL).bytes);
        assert!(v.inits.contains_key("sizes"));
        // Non-constant sizes: skipped.
        let g = G {
            nodes: vec![node("Split", "s", &["x", "dyn"], &["a", "b"], &[])],
            inputs: vec![value_info("x", DT_FLOAT, Some(&[8]))],
            ..G::default()
        };
        match rewrite(&model(13, &g, &[]), &RewriteOptions::ALL).unwrap() {
            Outcome::Unchanged { skipped } => assert!(skipped[0].contains("not a constant")),
            o => panic!("{o:?}"),
        }
    }

    #[test]
    fn split_opset_18_num_outputs() {
        // 10 into 3 parts: ceil chunks 4, 4, 2.
        let sp = node(
            "Split",
            "s",
            &["x"],
            &["a", "b", "c"],
            &[attr_i("axis", -1), attr_i("num_outputs", 3)],
        );
        let v = view(
            &rewritten(
                &single(18, sp, &["a", "b", "c"], &[2, 10]),
                &RewriteOptions::ALL,
            )
            .bytes,
        );
        let r: Vec<_> = v
            .nodes
            .iter()
            .map(|n| (v.i64s_of(&n.1[1])[0], v.i64s_of(&n.1[2])[0]))
            .collect();
        assert_eq!(r, [(0, 4), (4, 8), (8, 10)]);
        // From value_info rather than graph inputs.
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
        let v = view(&rewritten(&model(18, &g, &[]), &RewriteOptions::ALL).bytes);
        assert_eq!(v.ops(), ["Relu", "Slice", "Slice"]);
        assert_eq!(v.i64s_of(&v.nodes[2].1[1]), [3]);
        // No shape anywhere: skipped, logged.
        let sp = node(
            "Split",
            "s",
            &["x"],
            &["a", "b"],
            &[attr_i("num_outputs", 2)],
        );
        let g = G {
            nodes: vec![sp],
            inputs: vec![value_info("x", DT_FLOAT, None)],
            ..G::default()
        };
        match rewrite(&model(18, &g, &[]), &RewriteOptions::ALL).unwrap() {
            Outcome::Unchanged { skipped } => assert!(skipped[0].starts_with("Split 's'")),
            o => panic!("{o:?}"),
        }
    }

    /// BE-05: values parsed from the model near `i64::MAX` must skip the node with a reason, never
    /// overflow (debug: panic; release: wrapped Slice/Reshape bounds in the cache file).
    #[test]
    fn overflowing_split_and_flatten_values_are_skipped() {
        let skipped_of = |o: Outcome| match o {
            Outcome::Unchanged { skipped } => skipped,
            o => panic!("{o:?}"),
        };
        // Split(split=[i64::MAX, 1]) at opset 12 (attribute form, COREML options).
        let sp = node(
            "Split",
            "big",
            &["x"],
            &["a", "b"],
            &[attr_i("axis", 0), attr_ints("split", &[i64::MAX, 1], false)],
        );
        let s = skipped_of(
            rewrite(&single(12, sp, &["a", "b"], &[8]), &RewriteOptions::COREML).unwrap(),
        );
        assert_eq!(s.len(), 1);
        assert!(s[0].starts_with("Split 'big':"), "{s:?}");
        assert!(s[0].contains("overflow"), "{s:?}");
        // Opset 18 equal split of a dimension near i64::MAX: ceil(dim / k) overflows.
        let sp = node(
            "Split",
            "eq",
            &["x"],
            &["a", "b"],
            &[attr_i("axis", 0), attr_i("num_outputs", 2)],
        );
        let s = skipped_of(
            rewrite(
                &single(18, sp, &["a", "b"], &[i64::MAX]),
                &RewriteOptions::ALL,
            )
            .unwrap(),
        );
        assert!(
            s[0].starts_with("Split 'eq':") && s[0].contains("overflow"),
            "{s:?}"
        );
        // Flatten(axis=3) whose leading dimensions multiply past i64::MAX.
        let f = node("Flatten", "f", &["x"], &["y"], &[attr_i("axis", 3)]);
        let s = skipped_of(
            rewrite(
                &single(13, f, &["y"], &[i64::MAX, 2, 2, 5]),
                &RewriteOptions::ALL,
            )
            .unwrap(),
        );
        assert!(
            s[0].starts_with("Flatten 'f':") && s[0].contains("overflow"),
            "{s:?}"
        );
        // The same shapes without the overflow still rewrite (the guard is not over-eager).
        let f = node("Flatten", "f", &["x"], &["y"], &[attr_i("axis", 3)]);
        assert!(matches!(
            rewrite(&single(13, f, &["y"], &[3, 2, 2, 5]), &RewriteOptions::ALL).unwrap(),
            Outcome::Rewritten(_)
        ));
    }

    #[test]
    fn flatten_becomes_reshape() {
        let case = |axis: Option<i64>, dims: &[i64]| {
            let attrs: Vec<_> = axis.map(|a| attr_i("axis", a)).into_iter().collect();
            let f = node("Flatten", "f", &["x"], &["y"], &attrs);
            rewrite(&single(13, f, &["y"], dims), &RewriteOptions::ALL).unwrap()
        };
        let shape_of = |o: Outcome| match o {
            Outcome::Rewritten(r) => {
                let v = view(&r.bytes);
                assert_eq!(v.ops(), ["Reshape"]);
                assert_eq!(v.nodes[0].1[0], "x");
                assert_eq!(v.nodes[0].2, ["y"]);
                v.i64s_of(&v.nodes[0].1[1])
            }
            o => panic!("{o:?}"),
        };
        assert_eq!(shape_of(case(None, &[-1, 3, 4])), [0, -1]);
        let f = node("Flatten", "f", &["x"], &["y"], &[]);
        assert!(matches!(
            rewrite(&single(13, f, &["y"], &[1, 4]), &RewriteOptions::COREML).unwrap(),
            Outcome::Unchanged { .. }
        ));
        assert_eq!(shape_of(case(Some(0), &[-1, 3])), [1, -1]);
        assert_eq!(shape_of(case(Some(2), &[2, 3, 4, 5])), [6, -1]);
        assert_eq!(shape_of(case(Some(-1), &[2, 3, 4])), [6, -1]);
        assert_eq!(shape_of(case(Some(-2), &[-1, 3, 4])), [0, -1]);
        match case(Some(2), &[-1, 3, 4]) {
            Outcome::Unchanged { skipped } => assert!(skipped[0].contains("static leading")),
            o => panic!("{o:?}"),
        }
    }

    #[test]
    fn external_data_is_left_alone() {
        let ext = msg(|w| {
            w.int(TENSOR_DIMS, 2);
            w.int(TENSOR_DATA_TYPE, i64::from(DT_FLOAT));
            w.string(TENSOR_NAME, "w");
            let e = msg(|e| {
                e.string(ENTRY_KEY, "location");
                e.string(ENTRY_VALUE, "weights.bin");
            });
            w.bytes(TENSOR_EXTERNAL_DATA, &e);
            w.int(TENSOR_DATA_LOCATION, 1);
        });
        let g = G {
            nodes: vec![node("HardSigmoid", "hs", &["x"], &["y"], &[])],
            inits: vec![ext],
            inputs: vec![value_info("x", DT_FLOAT, Some(&[2]))],
            ..G::default()
        };
        assert!(matches!(
            rewrite(&model(13, &g, &[]), &RewriteOptions::ALL).unwrap(),
            Outcome::ExternalData
        ));
    }

    #[test]
    fn names_never_collide_and_other_domains_are_ignored() {
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
        let r = rewritten(&model(13, &g, &[]), &RewriteOptions::ALL);
        assert_eq!(
            r.counts.get("HardSigmoid"),
            Some(&1),
            "custom domain untouched"
        );
        let v = view(&r.bytes);
        assert_eq!(v.ops(), ["Relu", "Mul", "Add", "Clip", "HardSigmoid"]);
        assert!(!v.inits.contains_key("bop_rewrite/const_0"));
        assert!(v.inits.contains_key("bop_rewrite/const_0_1"));
    }

    #[test]
    fn rewrite_cached_writes_once_and_keeps_the_source() {
        let dir =
            std::env::temp_dir().join(format!("bop-onnx-rewrite-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("tiny.onnx");
        let m = single(
            12,
            node("HardSigmoid", "hs", &["x"], &["y"], &[]),
            &["y"],
            &[2],
        );
        std::fs::write(&src, &m).unwrap();
        let cache = dir.join("cache").join("coreml");
        let first = rewrite_cached(&src, &cache, &RewriteOptions::COREML).unwrap();
        let Cached::Rewritten {
            path,
            created,
            counts,
            ..
        } = first
        else {
            panic!("{first:?}")
        };
        assert!(created);
        assert_eq!(counts.get("HardSigmoid"), Some(&1));
        assert_eq!(path, cache.join(cache_file_name(&src, &sha256_hex(&m))));
        assert!(
            path.file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .starts_with("tiny-")
        );
        assert_eq!(std::fs::read(&src).unwrap(), m, "source untouched");
        let written = std::fs::read(&path).unwrap();
        assert_eq!(view(&written).ops(), ["Mul", "Add", "Clip"]);
        let again = rewrite_cached(&src, &cache, &RewriteOptions::COREML).unwrap();
        assert!(
            matches!(again, Cached::Rewritten { created: false, path: ref p, .. } if *p == path)
        );
        // No temporary files left behind.
        assert_eq!(std::fs::read_dir(&cache).unwrap().count(), 1);
        // A planted or corrupted file at the cache path is not trusted: it is replaced by the
        // real rewrite (BE-04).
        std::fs::write(&path, b"not the rewrite").unwrap();
        let replaced = rewrite_cached(&src, &cache, &RewriteOptions::COREML).unwrap();
        assert!(
            matches!(replaced, Cached::Rewritten { created: true, path: ref p, .. } if *p == path),
            "{replaced:?}"
        );
        assert_eq!(std::fs::read(&path).unwrap(), written);
        assert_eq!(std::fs::read_dir(&cache).unwrap().count(), 1);
        // A model with nothing to do is not copied.
        let plain = dir.join("plain.onnx");
        std::fs::write(
            &plain,
            single(12, node("Relu", "r", &["x"], &["y"], &[]), &["y"], &[2]),
        )
        .unwrap();
        assert!(matches!(
            rewrite_cached(&plain, &cache, &RewriteOptions::COREML).unwrap(),
            Cached::Original { .. }
        ));
        assert_eq!(std::fs::read_dir(&cache).unwrap().count(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn garbage_is_an_error() {
        assert!(rewrite(b"not a model at all \xff\xff", &RewriteOptions::COREML).is_err());
        assert!(rewrite(&[], &RewriteOptions::COREML).is_err(), "no graph");
    }

    /// Outputs of the rewrite before the lazy tensor parsing (BE-06), pinned: the change must
    /// not alter a single byte. Update only together with `REWRITER_VERSION`.
    #[test]
    fn golden_outputs_are_unchanged() {
        let pinned: &[(&str, &str)] = &[
            ("hs12", "R a77a31b4d2d8c1bf HardSigmoid x1 "),
            ("hs10", "R 0a9aee423c9838d0 HardSigmoid x1 "),
            ("hs6", "U HardSigmoid '<unnamed>': opset 6 < 7"),
            ("hsw17", "R 4cde96fbab2c29d7 HardSwish x1 "),
            ("hsw-f16", "R 35f089f09ee885d3 HardSwish x1 "),
            ("hsw-f64", "R 85a4545f628bf961 HardSwish x1 "),
            ("mixed", "U HardSigmoid 'hs': element type of 'h' unknown"),
            (
                "f16-weights",
                "U HardSigmoid 'hs': element type of 't' unknown",
            ),
            ("split12", "R 9a4b08e172ad8a0c Split x1 "),
            ("split12-packed", "R 8c1f347779b8b1e1 Split x1 "),
            ("split9", "R 95443490b5fd1cff Split x1 "),
            ("split11-eq", "R 9502f3a31b50ae9a Split x1 "),
            (
                "split11-sym",
                "U Split 's': equal split of 'x' needs its shape along axis 1",
            ),
            ("split13-raw", "R d6daa6509ebd2e4d Split x1 "),
            ("split13-typed", "R d6daa6509ebd2e4d Split x1 "),
            ("split13-const", "R d6daa6509ebd2e4d Split x1 "),
            ("split13-i32raw", "R 749801136f54bde7 Split x1 "),
            ("split13-i32typed", "R 749801136f54bde7 Split x1 "),
            ("split13-kept", "R 63f3a2d80367542b Split x1 "),
            ("split13-value-ints", "R f958a38bdf4fe867 Split x1 "),
            ("split18", "R 01c86dfa6bb9794c Split x1 "),
            ("split18-vi", "R f50d5ff7cc9beb25 Split x1 "),
            ("flat-none", "R 928df603336106b6 Flatten x1 "),
            ("flat0", "R f20b1485b1607b9b Flatten x1 "),
            ("flat2", "R dac212c2474a4045 Flatten x1 "),
            ("flat-1", "R fd01ba03dadb06c9 Flatten x1 "),
            (
                "flat2-sym",
                "U Flatten 'f': axis > 1 needs static leading dimensions",
            ),
            ("unknown-fields", "R fa1940d0e11b2b78 HardSigmoid x1 "),
            ("names", "R be84ea42ab2c6f2f HardSigmoid x1 "),
            ("external", "E"),
        ];
        let got: Vec<(&str, String)> = golden_corpus()
            .into_iter()
            .map(|(label, m, opts)| (label, outcome_digest(&rewrite(&m, &opts).unwrap())))
            .collect();
        assert_eq!(got.len(), pinned.len());
        for ((label, d), (pl, pd)) in got.iter().zip(pinned) {
            assert_eq!((*label, d.as_str()), (*pl, *pd));
        }
    }

    /// BE-06: weight tensors are scanned, not split into stored fields (before: 1,048,612
    /// stored `Field`s, ~40 MB, for this model), and the output is byte-identical to the rewrite
    /// before the change (length and digest pinned from the eager parser).
    #[test]
    fn big_initializers_are_not_materialized() {
        let (stored, r) = stored_fields(&big_model(true));
        let (without, _) = stored_fields(&big_model(false));
        // Each initializer costs only its own graph-level field.
        assert_eq!(
            stored,
            without + 2,
            "stored {stored}, without weights {without}"
        );
        assert!(stored < 64, "{stored}");
        assert_eq!(r.bytes.len(), 110_101_002);
        assert_eq!(
            sha256_hex(&r.bytes),
            "c7b411a6d052a4fd95bcf7ca7465f23927270b603500acb14878e0965a615d63"
        );
    }

    #[test]
    fn rewrite_size_limit() {
        assert_eq!(MAX_REWRITE_BYTES, 512 * 1024 * 1024);
        let dir = std::env::temp_dir().join(format!("bop-onnx-rewrite-cap-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("m.onnx");
        let m = single(
            12,
            node("HardSigmoid", "hs", &["x"], &["y"], &[]),
            &["y"],
            &[2],
        );
        std::fs::write(&src, &m).unwrap();
        let len = m.len() as u64;
        assert_eq!(read_capped(&src, len).unwrap().as_deref(), Some(&m[..]));
        assert_eq!(read_capped(&src, len - 1).unwrap(), None);
        let cache = dir.join("cache");
        // Over the limit: the original, nothing read into the cache.
        assert!(matches!(
            rewrite_cached_capped(&src, &cache, &RewriteOptions::COREML, len - 1).unwrap(),
            Cached::Original { ref skipped } if skipped.is_empty()
        ));
        assert!(!cache.exists());
        // At the limit: rewritten.
        assert!(matches!(
            rewrite_cached_capped(&src, &cache, &RewriteOptions::COREML, len).unwrap(),
            Cached::Rewritten { created: true, .. }
        ));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn file_equals_compares_in_chunks() {
        let dir = std::env::temp_dir().join(format!("bop-onnx-rewrite-eq-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("f");
        let data: Vec<u8> = (0..200_000u32).map(|i| (i % 251) as u8).collect();
        std::fs::write(&p, &data).unwrap();
        assert!(file_equals(&p, &data).unwrap());
        let mut other = data.clone();
        other[150_000] ^= 1;
        assert!(
            !file_equals(&p, &other).unwrap(),
            "difference in a later chunk"
        );
        assert!(!file_equals(&p, &data[..1000]).unwrap(), "length differs");
        assert!(file_equals(&dir.join("missing"), &data).is_err());
        std::fs::write(&p, b"").unwrap();
        assert!(file_equals(&p, b"").unwrap());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn cache_names() {
        let sha = "0123456789abcdef0123";
        assert_eq!(
            cache_file_name(Path::new("/m/IPcam-general.onnx"), sha),
            format!("IPcam-general-0123456789abcdef-v{REWRITER_VERSION}.onnx")
        );
    }
}
