//! Strip-and-retry for request fields Bedrock rejects.
//!
//! Bedrock validates the Anthropic request body more strictly than the Anthropic API, and Claude
//! Code sometimes sends fields it does not accept. Three are known:
//!
//! - `messages.N.output_config: Extra inputs are not permitted` (a top-level `output_config` is
//!   already dropped by the request translation, so only the per-message form reaches Bedrock)
//! - `thinking.adaptive.display: Input should be 'summarized', 'omitted'`
//! - `safeguards: Extra inputs are not permitted` (older profiles lack server-side review). The
//!   field and its `dangerous-tool-use-*` beta go together: the beta alone is rejected there too
//!   (`invalid beta flag`, which names no beta, so the beta net cannot catch it).
//!
//! The handler retries the request without exactly the fields Bedrock named, one error at a time.
//! Only the keys matched in `rejected_paths`, plus the top-level `safeguards`, are ever removed,
//! and only when the error names them, so an unrelated 400 is never rewritten.

use serde_json::Value;

use crate::translate::betas::is_safeguards_beta;

/// A rejected field that was removed from the request, e.g. `messages.1.output_config`.
#[derive(Debug, PartialEq, Eq)]
pub struct Stripped {
    pub bytes: Vec<u8>,
    /// Dotted paths of the removed fields, in the form Bedrock reported them.
    pub fields: Vec<String>,
    /// Betas removed alongside `safeguards`, so the caller can record them as unsupported.
    pub betas: Vec<String>,
}

/// Strip-and-retry rounds per request. Bedrock has been seen reporting one bad field per
/// rejection; `output_config` can sit on several messages, so the cap leaves room for a few of
/// them plus the thinking field. A request that still fails gets the original error.
pub const MAX_ROUNDS: usize = 5;

const EXTRA_INPUTS: &str = ": Extra inputs are not permitted";
const THINKING_DISPLAY: &str = "thinking.adaptive.display: Input should be";

/// Remove the fields named by a Bedrock `ValidationException` message from `body_bytes`.
///
/// Returns `None` when the error names none of the allowlisted fields, when the body cannot be
/// parsed, or when the named field is not present in the body (so a retry could not help).
/// `err` is the `Debug` form of the SDK error.
pub fn strip_rejected_fields(body_bytes: &[u8], err: &str) -> Option<Stripped> {
    let wanted = rejected_paths(err);
    let safeguards = names_safeguards(err);
    if wanted.is_empty() && !safeguards {
        return None;
    }
    let mut body: Value = serde_json::from_slice(body_bytes).ok()?;
    let mut fields = Vec::new();
    let mut betas = Vec::new();
    for path in wanted {
        if remove_path(&mut body, &path) {
            fields.push(path.reported);
        }
    }
    if safeguards
        && let Some(obj) = body.as_object_mut()
        && obj.remove("safeguards").is_some()
    {
        fields.push("safeguards".to_string());
        if let Some(arr) = obj.get_mut("anthropic_beta").and_then(|v| v.as_array_mut()) {
            arr.retain(|v| match v.as_str() {
                Some(b) if is_safeguards_beta(b) => {
                    betas.push(b.to_string());
                    false
                }
                _ => true,
            });
        }
    }
    if fields.is_empty() {
        return None;
    }
    Some(Stripped {
        bytes: serde_json::to_vec(&body).ok()?,
        fields,
        betas,
    })
}

/// True when the error says the top-level `safeguards` field is not accepted.
fn names_safeguards(err: &str) -> bool {
    err.match_indices(EXTRA_INPUTS)
        .any(|(at, _)| path_before(err, at) == "safeguards")
}

struct Target {
    /// Path as Bedrock reported it, for logs and metrics.
    reported: String,
    /// Where the key sits in the request body.
    segments: Vec<Seg>,
}

enum Seg {
    Key(&'static str),
    Index(usize),
}

fn rejected_paths(err: &str) -> Vec<Target> {
    let mut out: Vec<Target> = Vec::new();
    let mut push = |t: Target| {
        if !out.iter().any(|o| o.reported == t.reported) {
            out.push(t);
        }
    };

    // `output_config` at the top level or inside message N.
    for (at, _) in err.match_indices(EXTRA_INPUTS) {
        let path = path_before(err, at);
        if let Some(n) = path
            .strip_prefix("messages.")
            .and_then(|r| r.strip_suffix(".output_config"))
            .and_then(|n| n.parse::<usize>().ok())
        {
            push(Target {
                reported: path.to_string(),
                segments: vec![
                    Seg::Key("messages"),
                    Seg::Index(n),
                    Seg::Key("output_config"),
                ],
            });
        }
    }

    // Bedrock reports the union tag (`adaptive`) in the path; the body key is `thinking.display`.
    if field_boundary_match(err, THINKING_DISPLAY) {
        push(Target {
            reported: "thinking.adaptive.display".to_string(),
            segments: vec![Seg::Key("thinking"), Seg::Key("display")],
        });
    }
    out
}

/// The dotted path (`[A-Za-z0-9_.]`) that ends immediately before byte offset `end`.
fn path_before(s: &str, end: usize) -> &str {
    let head = &s[..end];
    let start = head
        .char_indices()
        .rev()
        .find(|(_, c)| !(c.is_ascii_alphanumeric() || *c == '_' || *c == '.'))
        .map(|(i, c)| i + c.len_utf8())
        .unwrap_or(0);
    &head[start..]
}

/// True when `needle` occurs in `s` and is not the tail of a longer path
/// (so `foo.thinking.adaptive.display` does not match).
fn field_boundary_match(s: &str, needle: &str) -> bool {
    s.match_indices(needle).any(|(i, _)| {
        s[..i]
            .chars()
            .next_back()
            .is_none_or(|c| !(c.is_ascii_alphanumeric() || c == '_' || c == '.'))
    })
}

/// Remove the key at `target`, returning whether anything was removed.
fn remove_path(body: &mut Value, target: &Target) -> bool {
    let (last, parents) = match target.segments.split_last() {
        Some(x) => x,
        None => return false,
    };
    let mut cur = body;
    for seg in parents {
        cur = match seg {
            Seg::Key(k) => match cur.get_mut(*k) {
                Some(v) => v,
                None => return false,
            },
            Seg::Index(i) => match cur.get_mut(*i) {
                Some(v) => v,
                None => return false,
            },
        };
    }
    match (last, cur.as_object_mut()) {
        (Seg::Key(k), Some(obj)) => obj.remove(*k).is_some(),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn run(body: &Value, err: &str) -> Option<(Value, Vec<String>)> {
        let s = strip_rejected_fields(&serde_json::to_vec(body).unwrap(), err)?;
        Some((serde_json::from_slice(&s.bytes).unwrap(), s.fields))
    }

    #[test]
    fn strips_message_output_config() {
        let body = json!({"messages":[{"role":"user","content":"a"},{"role":"user","content":"b","output_config":{"effort":"high"}}]});
        let err = "ValidationException { message: Some(\"messages.1.output_config: Extra inputs are not permitted\") }";
        let (out, fields) = run(&body, err).unwrap();
        assert_eq!(fields, vec!["messages.1.output_config"]);
        assert!(out["messages"][1].get("output_config").is_none());
        assert_eq!(out["messages"][1]["content"], "b");
        assert_eq!(out["messages"][0], body["messages"][0]);
    }

    #[test]
    fn strips_safeguards_and_its_beta_together() {
        let body = json!({"max_tokens":10,"safeguards":[{"type":"dangerous_tool_use"}],
            "anthropic_beta":["claude-code-20250219","dangerous-tool-use-2026-09-03"]});
        let err =
            "ValidationException { message: Some(\"safeguards: Extra inputs are not permitted\") }";
        let s = strip_rejected_fields(&serde_json::to_vec(&body).unwrap(), err).unwrap();
        let out: Value = serde_json::from_slice(&s.bytes).unwrap();
        assert_eq!(s.fields, vec!["safeguards"]);
        assert_eq!(s.betas, vec!["dangerous-tool-use-2026-09-03"]);
        assert!(out.get("safeguards").is_none());
        assert_eq!(out["anthropic_beta"], json!(["claude-code-20250219"]));
        assert_eq!(out["max_tokens"], 10);
    }

    #[test]
    fn safeguards_error_with_no_safeguards_in_body_is_not_a_retry() {
        let body = json!({"max_tokens":10});
        let err = "safeguards: Extra inputs are not permitted";
        assert!(strip_rejected_fields(&serde_json::to_vec(&body).unwrap(), err).is_none());
    }

    #[test]
    fn nested_safeguards_path_is_not_the_top_level_field() {
        let body = json!({"safeguards":[{}],"anthropic_beta":["dangerous-tool-use-2026-09-03"]});
        let err = "messages.0.safeguards: Extra inputs are not permitted";
        assert!(strip_rejected_fields(&serde_json::to_vec(&body).unwrap(), err).is_none());
    }

    #[test]
    fn strips_thinking_display_and_keeps_the_rest() {
        let body = json!({"thinking":{"type":"adaptive","display":"full"},"max_tokens":10});
        let err =
            "message: Some(\"thinking.adaptive.display: Input should be 'summarized', 'omitted'\")";
        let (out, fields) = run(&body, err).unwrap();
        assert_eq!(fields, vec!["thinking.adaptive.display"]);
        assert_eq!(out["thinking"], json!({"type":"adaptive"}));
        assert_eq!(out["max_tokens"], 10);
    }

    #[test]
    fn strips_both_when_one_message_names_both() {
        let body =
            json!({"thinking":{"type":"adaptive","display":"x"},"messages":[{"output_config":{}}]});
        let err = "messages.0.output_config: Extra inputs are not permitted; thinking.adaptive.display: Input should be 'summarized', 'omitted'";
        let (out, fields) = run(&body, err).unwrap();
        assert_eq!(fields.len(), 2);
        assert!(out["thinking"].get("display").is_none());
        assert!(out["messages"][0].get("output_config").is_none());
    }

    #[test]
    fn fields_named_one_at_a_time_are_stripped_in_turn() {
        let body =
            json!({"thinking":{"type":"adaptive","display":"x"},"messages":[{"output_config":{}}]});
        let first = "messages.0.output_config: Extra inputs are not permitted";
        let second = "thinking.adaptive.display: Input should be 'summarized', 'omitted'";
        let s1 = strip_rejected_fields(&serde_json::to_vec(&body).unwrap(), first).unwrap();
        assert_eq!(s1.fields, vec!["messages.0.output_config"]);
        let s2 = strip_rejected_fields(&s1.bytes, second).unwrap();
        let out: Value = serde_json::from_slice(&s2.bytes).unwrap();
        assert!(out["thinking"].get("display").is_none());
        assert!(out["messages"][0].get("output_config").is_none());
        // Nothing left to strip: the loop ends instead of retrying forever.
        assert!(strip_rejected_fields(&s2.bytes, second).is_none());
    }

    #[test]
    fn other_fields_are_never_stripped() {
        let body = json!({"foo":1,"tools":[{"defer_loading":true}]});
        for err in [
            "foo: Extra inputs are not permitted",
            "tools.0.custom.defer_loading: Extra inputs are not permitted",
            "messages.0.cache_control: Extra inputs are not permitted",
            "x.output_config: Extra inputs are not permitted",
            "messages.0.not_output_config: Extra inputs are not permitted",
            "messages.x.output_config: Extra inputs are not permitted",
            "output_config: Extra inputs are not permitted",
            "other.thinking.adaptive.display: Input should be 'summarized'",
        ] {
            assert!(run(&body, err).is_none(), "{err}");
        }
    }

    #[test]
    fn returns_none_when_the_field_is_absent_so_no_retry_loop() {
        let body = json!({"messages":[{"role":"user","content":"a"}]});
        assert!(
            run(
                &body,
                "messages.0.output_config: Extra inputs are not permitted"
            )
            .is_none()
        );
        assert!(
            run(
                &body,
                "messages.9.output_config: Extra inputs are not permitted"
            )
            .is_none()
        );
        // Second pass over an already-stripped body finds nothing.
        let body = json!({"messages":[{"output_config":{}}]});
        let s = strip_rejected_fields(
            &serde_json::to_vec(&body).unwrap(),
            "messages.0.output_config: Extra inputs are not permitted",
        )
        .unwrap();
        assert!(
            strip_rejected_fields(
                &s.bytes,
                "messages.0.output_config: Extra inputs are not permitted"
            )
            .is_none()
        );
    }

    #[test]
    fn unparseable_body_or_error_gives_none() {
        assert!(
            strip_rejected_fields(
                b"not json",
                "messages.0.output_config: Extra inputs are not permitted"
            )
            .is_none()
        );
        assert!(strip_rejected_fields(b"{}", "service error").is_none());
    }

    #[test]
    fn matches_the_real_sdk_debug_format() {
        use aws_sdk_bedrockruntime::error::SdkError;
        use aws_sdk_bedrockruntime::operation::invoke_model_with_response_stream::InvokeModelWithResponseStreamError;
        use aws_sdk_bedrockruntime::types::error::ValidationException;

        let exc = ValidationException::builder()
            .message("messages.1.output_config: Extra inputs are not permitted, thinking.adaptive.display: Input should be 'summarized', 'omitted'")
            .build();
        let err: SdkError<InvokeModelWithResponseStreamError, ()> = SdkError::service_error(
            InvokeModelWithResponseStreamError::ValidationException(exc),
            (),
        );
        let body = json!({"thinking":{"type":"adaptive","display":"x"},"messages":[{"role":"user"},{"output_config":{}}]});
        let (out, fields) = run(&body, &format!("{err:?}")).unwrap();
        assert_eq!(
            fields,
            vec!["messages.1.output_config", "thinking.adaptive.display"]
        );
        assert!(out["thinking"].get("display").is_none());
    }

    #[test]
    fn non_ascii_text_before_the_path_does_not_panic() {
        let body = json!({"messages":[{"output_config":{}}]});
        for pre in ["\u{201c}", "é", "日本語 ", "\u{1f600}"] {
            let err = format!("{pre}messages.0.output_config: Extra inputs are not permitted");
            let (out, _) = run(&body, &err).unwrap();
            assert!(out["messages"][0].get("output_config").is_none());
        }
    }

    #[test]
    fn odd_body_shapes_are_left_alone() {
        let err = "thinking.adaptive.display: Input should be 'summarized', 'omitted'";
        assert!(run(&json!({"thinking":"adaptive"}), err).is_none());
        assert!(run(&json!({"thinking":{"type":"adaptive"}}), err).is_none());
        let oc = "messages.5.output_config: Extra inputs are not permitted";
        assert!(run(&json!({"messages":[{"output_config":{}}]}), oc).is_none());
        assert!(
            run(
                &json!({"messages":["text"]}),
                "messages.0.output_config: Extra inputs are not permitted"
            )
            .is_none()
        );
    }

    #[test]
    fn a_field_named_twice_is_reported_once() {
        let body = json!({"messages":[{"output_config":{}}]});
        let err = "messages.0.output_config: Extra inputs are not permitted; messages.0.output_config: Extra inputs are not permitted";
        let (_, fields) = run(&body, err).unwrap();
        assert_eq!(fields, vec!["messages.0.output_config"]);
    }
}
