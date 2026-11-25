//! Byte caps on UTF-8 text (std only): [`cap_bytes`] cuts a string to at most N bytes on a
//! character boundary, [`cap_name`] to [`MAX_NAME_BYTES`] for a client-chosen name (an upload's
//! `image_name`, a static-memory camera key). Callers that cut by characters (an ellipsis for
//! display) keep their own helpers: their limit counts something else.

/// Longest client-chosen name kept, in bytes: an upload's `image_name` (cut at the request edge)
/// and a [`crate::static_memory::camera_key`]. Both land in every dataset sidecar and in memory
/// per camera; 255 bytes is what most file systems take as one file name.
pub const MAX_NAME_BYTES: usize = 255;

/// `s` cut to at most [`MAX_NAME_BYTES`] bytes, on a character boundary.
pub fn cap_name(s: &str) -> &str {
    cap_bytes(s, MAX_NAME_BYTES)
}

/// `s` cut to at most `max` bytes, on a character boundary.
pub fn cap_bytes(s: &str, max: usize) -> &str {
    if s.len() <= max {
        return s;
    }
    let mut end = max;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    s.get(..end).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// M2: a client-chosen name cannot grow what is kept per camera and written per frame.
    #[test]
    fn names_are_capped_on_a_char_boundary() {
        assert_eq!(cap_name("short.jpg"), "short.jpg");
        let ascii = "a".repeat(1000);
        assert_eq!(cap_name(&ascii).len(), MAX_NAME_BYTES);
        // 'é' is two bytes: 255 falls inside the 128th, so the cut is at 254.
        let wide = "é".repeat(200);
        let cut = cap_name(&wide);
        assert_eq!(cut.len(), 254);
        assert!(cut.chars().all(|c| c == 'é'));
        assert_eq!(cap_bytes("日本", 4), "日");
        assert_eq!(cap_bytes("日本", 2), "");
    }

    #[test]
    fn cap_bytes_keeps_short_text_and_cuts_at_the_limit() {
        assert_eq!(cap_bytes("", 0), "");
        assert_eq!(cap_bytes("abc", 0), "");
        assert_eq!(cap_bytes("abc", 3), "abc");
        assert_eq!(cap_bytes("abcd", 3), "abc");
        // A four-byte character is kept whole or not at all.
        assert_eq!(cap_bytes("a😀", 4), "a");
        assert_eq!(cap_bytes("a😀", 5), "a😀");
    }
}
