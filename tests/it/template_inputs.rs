//! Lint of every form field in `templates/` (plain string scanning, askama `{% %}` / `{{ }}`
//! inside a tag are skipped), so the phone keyboards stay right:
//!
//! (a) a `type="number"` field whose step is absent or a whole number has `inputmode="numeric"`
//!     (it only takes integers); one with a fractional or `any` step has none (the iOS decimal
//!     keypad shows the locale's separator, which `type=number` may refuse);
//! (b) a text-like field (`text`, `url`, `search`, `textarea`) whose `name` or `id` has a word in
//!     `CODE_WORDS` (paths, folders, hosts, cameras, classes, models, URLs, names, filter
//!     expressions, the learning rate: values a phone must not capitalize or correct) has
//!     `autocapitalize="none"`, `autocorrect="off"` and `spellcheck="false"`. Words are the pieces
//!     of the name split at `_ - .`, so `username` (which sets its own attributes) and prose
//!     fields such as `license` or `notes` are not caught;
//! (c) no template says `one-time-code` (the setup code is typed by hand from the console).
#![allow(clippy::unwrap_used)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

const CODE_WORDS: &[&str] = &[
    "path", "dir", "folder", "host", "hosts", "camera", "cameras", "class", "classes", "model",
    "models", "url", "root", "name", "label", "labels", "frames", "filter", "days", "bind", "lr",
];

struct Tag {
    file: String,
    kind: String,
    attrs: BTreeMap<String, String>,
}

fn files(dir: &Path, out: &mut Vec<PathBuf>) {
    for e in std::fs::read_dir(dir).unwrap() {
        let p = e.unwrap().path();
        if p.is_dir() {
            files(&p, out);
        } else if p.extension().is_some_and(|x| x == "html") {
            out.push(p);
        }
    }
}

/// Every `<input ...>` / `<textarea ...>` start tag of `text` with its attributes (lower-cased
/// names; a bare attribute has an empty value).
fn tags(file: &str, text: &str) -> Vec<Tag> {
    let mut out = Vec::new();
    let b = text.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if !text.is_char_boundary(i) {
            i += 1;
            continue;
        }
        let rest = &text[i..];
        let kind = ["input", "textarea"].into_iter().find(|k| {
            rest.starts_with(&format!("<{k}"))
                && rest[k.len() + 1..]
                    .chars()
                    .next()
                    .is_some_and(|c| c.is_whitespace() || c == '>')
        });
        let Some(kind) = kind else {
            i += 1;
            continue;
        };
        // Find the closing `>` outside quotes and outside askama blocks.
        let mut j = i + kind.len() + 1;
        let mut body = String::new();
        let mut quote = None;
        while j < b.len() {
            let r = &text[j..];
            if quote.is_none() && (r.starts_with("{%") || r.starts_with("{{")) {
                let close = if r.starts_with("{%") { "%}" } else { "}}" };
                j += r.find(close).map_or(r.len(), |p| p + 2);
                body.push(' ');
                continue;
            }
            let c = r.chars().next().unwrap();
            match (quote, c) {
                (None, '>') => break,
                (None, '"' | '\'') => quote = Some(c),
                (Some(q), c) if c == q => quote = None,
                _ => {}
            }
            body.push(c);
            j += c.len_utf8();
        }
        out.push(Tag {
            file: file.to_string(),
            kind: kind.to_string(),
            attrs: parse_attrs(&body),
        });
        i = j;
    }
    out
}

fn parse_attrs(body: &str) -> BTreeMap<String, String> {
    let mut m = BTreeMap::new();
    let c: Vec<char> = body.chars().collect();
    let mut i = 0;
    while i < c.len() {
        if c[i].is_whitespace() || c[i] == '/' {
            i += 1;
            continue;
        }
        let s = i;
        while i < c.len() && !c[i].is_whitespace() && c[i] != '=' && c[i] != '/' {
            i += 1;
        }
        let name: String = c[s..i].iter().collect::<String>().to_lowercase();
        let mut value = String::new();
        let mut k = i;
        while k < c.len() && c[k].is_whitespace() {
            k += 1;
        }
        if k < c.len() && c[k] == '=' {
            k += 1;
            while k < c.len() && c[k].is_whitespace() {
                k += 1;
            }
            if k < c.len() && (c[k] == '"' || c[k] == '\'') {
                let q = c[k];
                k += 1;
                while k < c.len() && c[k] != q {
                    value.push(c[k]);
                    k += 1;
                }
                k += 1;
            } else {
                while k < c.len() && !c[k].is_whitespace() {
                    value.push(c[k]);
                    k += 1;
                }
            }
            i = k;
        }
        m.insert(name, value);
    }
    m
}

fn all_tags() -> Vec<Tag> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("templates");
    let mut fs = Vec::new();
    files(&root, &mut fs);
    assert!(fs.len() > 10, "templates not found under {root:?}");
    let mut out = Vec::new();
    for f in fs {
        let text = std::fs::read_to_string(&f).unwrap();
        let rel = f.strip_prefix(&root).unwrap().display().to_string();
        out.extend(tags(&rel, &text));
    }
    out
}

fn describe(t: &Tag) -> String {
    format!(
        "{}: <{} name={:?} id={:?}>",
        t.file,
        t.kind,
        t.attrs.get("name"),
        t.attrs.get("id")
    )
}

fn whole_step(step: Option<&String>) -> bool {
    step.is_none_or(|s| !s.is_empty() && s.chars().all(|c| c.is_ascii_digit()))
}

fn number_problems(ts: &[Tag]) -> Vec<String> {
    let mut bad = Vec::new();
    for t in ts
        .iter()
        .filter(|t| t.attrs.get("type").map(String::as_str) == Some("number"))
    {
        let numeric = t.attrs.get("inputmode").map(String::as_str) == Some("numeric");
        let has = t.attrs.contains_key("inputmode");
        if whole_step(t.attrs.get("step")) {
            if !numeric {
                bad.push(format!(
                    "whole-number field without inputmode=\"numeric\": {}",
                    describe(t)
                ));
            }
        } else if has {
            bad.push(format!("decimal field with an inputmode: {}", describe(t)));
        }
    }
    bad
}

fn text_problems(ts: &[Tag]) -> Vec<String> {
    let mut bad = Vec::new();
    for t in ts {
        let ty = t.attrs.get("type").map_or("text", String::as_str);
        if !(t.kind == "textarea" || matches!(ty, "text" | "url" | "search")) {
            continue;
        }
        let ident = format!(
            "{} {}",
            t.attrs.get("name").map_or("", String::as_str),
            t.attrs.get("id").map_or("", String::as_str)
        )
        .to_lowercase();
        // A name made only of askama output has no words to judge.
        if !ident
            .split(|c: char| !c.is_ascii_alphanumeric())
            .any(|w| CODE_WORDS.contains(&w))
        {
            continue;
        }
        for (k, v) in [
            ("autocapitalize", "none"),
            ("autocorrect", "off"),
            ("spellcheck", "false"),
        ] {
            if t.attrs.get(k).map(String::as_str) != Some(v) {
                bad.push(format!("missing {k}=\"{v}\": {}", describe(t)));
            }
        }
    }
    bad
}

#[test]
fn whole_number_fields_ask_for_the_numeric_keypad_and_decimals_do_not() {
    let ts = all_tags();
    assert!(
        ts.iter()
            .filter(|t| t.attrs.get("type").is_some_and(|v| v == "number"))
            .count()
            > 30
    );
    let bad = number_problems(&ts);
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}

#[test]
fn path_host_model_camera_class_fields_are_not_capitalized_or_corrected() {
    let ts = all_tags();
    let bad = text_problems(&ts);
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}

#[test]
fn no_template_asks_for_a_one_time_code() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("templates");
    let mut fs = Vec::new();
    files(&root, &mut fs);
    for f in fs {
        let text = std::fs::read_to_string(&f).unwrap();
        assert!(!text.contains("one-time-code"), "{f:?}");
    }
}

// The scanner and the three rules can fail: these feed them broken markup.
#[test]
fn the_rules_catch_broken_markup() {
    let broken = r#"
      <input name="port" type="number" min="1" max="65535">
      <input type="number" name="t" step="0.01" inputmode="numeric">
      <input
        id="x" value="{% if a > b %}1{% endif %}" type="number" step="32">
      <input type="text" name="log_path" value="{{ p }}">
      <textarea name="models_json" autocapitalize="none"></textarea>
      <input type="text" name="license">
      <input type="number" name="ok" inputmode="numeric" min="0">
    "#;
    let ts = tags("t.html", broken);
    assert_eq!(ts.len(), 7);
    assert_eq!(number_problems(&ts).len(), 3, "{:?}", number_problems(&ts));
    let text = text_problems(&ts);
    // log_path: all three missing; models_json: corrector and spellcheck missing; license: ignored.
    assert_eq!(text.len(), 5, "{text:?}");
    // Attributes in another order and an askama block inside the tag still parse.
    let t = tags(
        "t.html",
        r#"<input {% if x %}disabled{% endif %} spellcheck="false" name="dir" autocorrect="off" autocapitalize="none" type="text">"#,
    );
    assert!(text_problems(&t).is_empty());
}
