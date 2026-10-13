//! Protobuf wire format: just enough to split a message into fields, re-emit untouched fields
//! byte for byte (`Field::raw`) and encode the few field kinds the rewrite writes (`Writer`).
//! Groups (wire types 3/4) are rejected; getters follow protobuf's "last occurrence wins" and
//! return the proto default when a field is absent.

use anyhow::{Context, Result, bail};

/// A decoded field value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Value<'a> {
    Varint(u64),
    Fixed64(u64),
    Len(&'a [u8]),
    Fixed32(u32),
}

/// One field of a message: number, value and its complete encoding (key included), so untouched
/// fields can be written back unchanged.
#[derive(Debug, Clone, Copy)]
pub(super) struct Field<'a> {
    pub num: u32,
    pub value: Value<'a>,
    pub raw: &'a [u8],
}

pub(super) fn read_varint(buf: &[u8], pos: &mut usize) -> Result<u64> {
    let mut v = 0u64;
    for shift in (0..64).step_by(7) {
        let Some(&b) = buf.get(*pos) else {
            bail!("truncated varint at byte {pos}")
        };
        *pos += 1;
        v |= u64::from(b & 0x7f) << shift;
        if b & 0x80 == 0 {
            return Ok(v);
        }
    }
    bail!("varint longer than 10 bytes")
}

/// Split a message into its fields (groups, wire types 3/4, are not used by ONNX and rejected).
///
/// Stores one [`Field`] (40 bytes) per encoded field, so only for messages with few fields
/// (model, graph, node, value info); weight tensors and attributes go through [`scan`] (BE-06).
pub(super) fn parse_fields(buf: &[u8]) -> Result<Vec<Field<'_>>> {
    let mut out = Vec::new();
    scan(buf, |f| {
        out.push(f);
        Ok(())
    })?;
    #[cfg(test)]
    super::test_models::FIELDS_STORED.with(|c| c.set(c.get() + out.len()));
    Ok(out)
}

/// Visit the fields of a message in order without storing them (same decoding and errors as
/// [`parse_fields`]).
pub(super) fn scan<'a>(
    buf: &'a [u8],
    mut visit: impl FnMut(Field<'a>) -> Result<()>,
) -> Result<()> {
    let mut pos = 0;
    while pos < buf.len() {
        let start = pos;
        let key = read_varint(buf, &mut pos)?;
        let num = u32::try_from(key >> 3)
            .ok()
            .filter(|n| *n != 0)
            .with_context(|| format!("invalid field number at byte {start}"))?;
        let value = match key & 7 {
            0 => Value::Varint(read_varint(buf, &mut pos)?),
            1 => {
                let b = buf
                    .get(pos..)
                    .and_then(|r| r.first_chunk::<8>())
                    .context("truncated fixed64")?;
                pos += 8;
                Value::Fixed64(u64::from_le_bytes(*b))
            }
            2 => {
                let n = usize::try_from(read_varint(buf, &mut pos)?)?;
                let end = pos
                    .checked_add(n)
                    .filter(|e| *e <= buf.len())
                    .context("truncated length-delimited field")?;
                let v = buf
                    .get(pos..end)
                    .context("truncated length-delimited field")?;
                pos = end;
                Value::Len(v)
            }
            5 => {
                let b = buf
                    .get(pos..)
                    .and_then(|r| r.first_chunk::<4>())
                    .context("truncated fixed32")?;
                pos += 4;
                Value::Fixed32(u32::from_le_bytes(*b))
            }
            wt => bail!("unsupported wire type {wt} (field {num})"),
        };
        let raw = buf
            .get(start..pos)
            .context("field extends past the end of the message")?;
        visit(Field { num, value, raw })?;
    }
    Ok(())
}

/// Protobuf encoder for the few field kinds the rewrite emits.
#[derive(Debug, Default)]
pub(super) struct Writer {
    pub(super) buf: Vec<u8>,
}

impl Writer {
    #[allow(
        clippy::cast_possible_truncation,
        reason = "only the low 7 bits of each group are written; the loop keeps the rest"
    )]
    pub(super) fn varint(&mut self, mut v: u64) {
        while v >= 0x80 {
            self.buf.push((v as u8) | 0x80);
            v >>= 7;
        }
        self.buf.push(v as u8);
    }
    pub(super) fn key(&mut self, num: u32, wire_type: u64) {
        self.varint((u64::from(num) << 3) | wire_type);
    }
    pub(super) fn bytes(&mut self, num: u32, b: &[u8]) {
        self.key(num, 2);
        self.varint(b.len() as u64);
        self.buf.extend_from_slice(b);
    }
    pub(super) fn string(&mut self, num: u32, s: &str) {
        self.bytes(num, s.as_bytes());
    }
    /// int32/int64/enum: negative values are sign-extended to 10 bytes, as protobuf does.
    #[allow(
        clippy::cast_sign_loss,
        reason = "protobuf encodes int64 as its two's-complement bit pattern"
    )]
    pub(super) fn int(&mut self, num: u32, v: i64) {
        self.key(num, 0);
        self.varint(v as u64);
    }
    pub(super) fn float(&mut self, num: u32, v: f32) {
        self.key(num, 5);
        self.buf.extend_from_slice(&v.to_bits().to_le_bytes());
    }
    pub(super) fn raw(&mut self, b: &[u8]) {
        self.buf.extend_from_slice(b);
    }
}

pub(super) fn len_of<'a>(f: &Field<'a>) -> Option<&'a [u8]> {
    match f.value {
        Value::Len(b) => Some(b),
        _ => None,
    }
}

pub(super) fn as_str(b: &[u8]) -> Result<&str> {
    std::str::from_utf8(b).context("string field is not UTF-8")
}

/// Last occurrence of a string field ("" when absent, as protobuf defaults).
pub(super) fn get_str<'a>(fields: &[Field<'a>], num: u32) -> Result<&'a str> {
    match fields.iter().rev().find(|f| f.num == num).and_then(len_of) {
        Some(b) => as_str(b),
        None => Ok(""),
    }
}

pub(super) fn get_len<'a>(fields: &[Field<'a>], num: u32) -> Option<&'a [u8]> {
    fields.iter().rev().find(|f| f.num == num).and_then(len_of)
}

pub(super) fn get_int(fields: &[Field<'_>], num: u32) -> Option<i64> {
    fields
        .iter()
        .rev()
        .find_map(|f| match (f.num == num, f.value) {
            (true, Value::Varint(v)) => Some(v as i64),
            _ => None,
        })
}

pub(super) fn rep_str<'a>(fields: &[Field<'a>], num: u32) -> Result<Vec<&'a str>> {
    fields
        .iter()
        .filter(|f| f.num == num)
        .filter_map(len_of)
        .map(as_str)
        .collect()
}

/// The integers of one occurrence of a repeated int32/int64 field (packed or not).
fn int_values(f: &Field<'_>, out: &mut impl FnMut(i64)) -> Result<()> {
    match f.value {
        Value::Varint(v) => out(v as i64),
        Value::Len(b) => {
            let mut pos = 0;
            while pos < b.len() {
                out(read_varint(b, &mut pos)? as i64);
            }
        }
        _ => bail!("field {}: unexpected wire type for an integer", f.num),
    }
    Ok(())
}

/// Repeated int32/int64 field `num` of the message `msg`, read without storing its fields.
/// `out` sees every value in order.
pub(super) fn scan_ints(msg: &[u8], num: u32, mut out: impl FnMut(i64)) -> Result<()> {
    scan(msg, |f| {
        if f.num == num {
            int_values(&f, &mut out)?;
        }
        Ok(())
    })
}

/// [`scan_ints`] collected (for the few values the rewrite reads).
pub(super) fn collect_ints(msg: &[u8], num: u32) -> Result<Vec<i64>> {
    let mut v = Vec::new();
    scan_ints(msg, num, |x| v.push(x))?;
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn varints_round_trip() {
        for v in [0u64, 1, 127, 128, 300, 1 << 35, u64::MAX] {
            let mut w = Writer::default();
            w.varint(v);
            let mut pos = 0;
            assert_eq!(read_varint(&w.buf, &mut pos).unwrap(), v);
            assert_eq!(pos, w.buf.len());
        }
        assert!(read_varint(&[0x80], &mut 0).is_err());
        assert!(read_varint(&[0xff; 11], &mut 0).is_err());
    }

    #[test]
    fn negative_ints_are_ten_bytes() {
        let mut w = Writer::default();
        w.int(3, -1);
        assert_eq!(w.buf.len(), 11);
        let f = parse_fields(&w.buf).unwrap();
        assert_eq!(get_int(&f, 3), Some(-1));
    }

    #[test]
    fn truncated_messages_are_errors() {
        assert!(parse_fields(&[0x0a, 0x05, 1, 2]).is_err());
        assert!(parse_fields(&[0x0d, 1, 2]).is_err());
        assert!(parse_fields(&[0x0b]).is_err(), "groups are rejected");
        assert!(parse_fields(&[0x00, 0x00]).is_err(), "field number 0");
    }
}
