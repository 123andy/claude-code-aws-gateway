//! Cache-aware beta filtering and Bedrock ValidationException parsing.

use serde_json::Value;

use crate::endpoint::EndpointClient;

/// Prefix of the beta that enables server-side review of risky tool calls (auto mode). The date
/// suffix changes between Claude Code releases, so match the prefix.
pub const SAFEGUARDS_BETA_PREFIX: &str = "dangerous-tool-use-";

pub fn is_safeguards_beta(beta: &str) -> bool {
    beta.starts_with(SAFEGUARDS_BETA_PREFIX)
}

/// The `dangerous-tool-use-*` betas in the request's `anthropic-beta` header(s). Claude Code sends
/// its betas in that header, not in the body. Only this family is read from the header: forwarding
/// the rest would hit "invalid beta flag" (which names no beta) on older profiles.
pub fn safeguards_betas_from_header(headers: &axum::http::HeaderMap) -> Vec<String> {
    headers
        .get_all("anthropic-beta")
        .iter()
        .filter_map(|v| v.to_str().ok())
        .flat_map(|v| v.split(','))
        .map(str::trim)
        .filter(|b| is_safeguards_beta(b))
        .map(String::from)
        .collect()
}

/// Keep `safeguards` only while its beta is in `betas`. Bedrock rejects the field without the
/// beta ("Extra inputs are not permitted"), so the two must travel together.
pub fn pair_safeguards(betas: &[String], safeguards: &mut Option<Value>) {
    if !betas.iter().any(|b| is_safeguards_beta(b)) {
        *safeguards = None;
    }
}

/// Extract beta names from a Bedrock `ValidationException` error message.
///
/// Scans `error_message` for any substring in `candidates` (case-insensitive).
/// Returns the matched candidates — these are the betas Bedrock rejected.
///
/// Intentionally uses substring search rather than a regex so the logic is
/// robust to minor Bedrock error-message format drift.
pub fn parse_rejected_betas(error_message: &str, candidates: &[String]) -> Vec<String> {
    let lower = error_message.to_lowercase();
    candidates
        .iter()
        .filter(|c| lower.contains(&c.to_lowercase()))
        .cloned()
        .collect()
}

/// Result of filtering a beta list through the per-endpoint capability cache.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BetaFilterResult {
    /// Betas to include in the forwarded request (cache says supported OR unknown).
    pub kept: Vec<String>,
    /// Betas dropped because the cache recorded `Some(false)` for this endpoint+profile.
    pub dropped: Vec<String>,
    /// Subset of `kept` whose cache state was `None` (absent or expired) at filter time.
    /// On a successful response these should be opportunistically marked as supported.
    pub unknown: Vec<String>,
}

/// Filter `incoming` betas through the per-endpoint capability cache.
///
/// - `Some(true)`  → kept, not unknown
/// - `Some(false)` → dropped
/// - `None`        → kept AND placed in `unknown` (optimistic; learn on success)
pub async fn filter_betas_by_cache(
    client: &EndpointClient,
    profile: &str,
    incoming: &[String],
) -> BetaFilterResult {
    let mut kept = Vec::new();
    let mut dropped = Vec::new();
    let mut unknown = Vec::new();

    for b in incoming {
        match client.is_beta_supported(profile, b).await {
            Some(true) => kept.push(b.clone()),
            Some(false) => dropped.push(b.clone()),
            None => {
                kept.push(b.clone());
                unknown.push(b.clone());
            }
        }
    }

    BetaFilterResult {
        kept,
        dropped,
        unknown,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidates() -> Vec<String> {
        vec![
            "context-1m-2025-08-07".to_string(),
            "interleaved-thinking-2025-05-14".to_string(),
            "claude-code-x".to_string(),
        ]
    }

    // ── safeguards_betas_from_header ──────────────────────────────────────────

    #[test]
    fn header_betas_keep_only_the_safeguards_family() {
        let mut h = axum::http::HeaderMap::new();
        h.append(
            "anthropic-beta",
            "claude-code-20250219, dangerous-tool-use-2026-09-03,effort-2025-11-24"
                .parse()
                .unwrap(),
        );
        h.append("anthropic-beta", "afk-mode-2026-01-31".parse().unwrap());
        assert_eq!(
            safeguards_betas_from_header(&h),
            vec!["dangerous-tool-use-2026-09-03".to_string()]
        );
    }

    #[test]
    fn no_header_means_no_betas() {
        assert!(safeguards_betas_from_header(&axum::http::HeaderMap::new()).is_empty());
    }

    // ── pair_safeguards ───────────────────────────────────────────────────────

    #[test]
    fn safeguards_kept_when_beta_present() {
        let mut sg = Some(serde_json::json!([{"type": "dangerous_tool_use"}]));
        pair_safeguards(
            &[
                "claude-code-20250219".into(),
                "dangerous-tool-use-2026-09-03".into(),
            ],
            &mut sg,
        );
        assert!(sg.is_some());
    }

    #[test]
    fn safeguards_kept_for_a_future_dated_beta() {
        let mut sg = Some(serde_json::json!([{}]));
        pair_safeguards(&["dangerous-tool-use-2027-01-01".into()], &mut sg);
        assert!(sg.is_some());
    }

    #[test]
    fn safeguards_dropped_when_beta_absent() {
        let mut sg = Some(serde_json::json!([{"type": "dangerous_tool_use"}]));
        pair_safeguards(&["claude-code-20250219".into()], &mut sg);
        assert!(sg.is_none());
        let mut sg = Some(serde_json::json!([{}]));
        pair_safeguards(&[], &mut sg);
        assert!(sg.is_none());
    }

    #[test]
    fn pairing_none_stays_none() {
        let mut sg: Option<Value> = None;
        pair_safeguards(&["dangerous-tool-use-2026-09-03".into()], &mut sg);
        assert!(sg.is_none());
    }

    // ── parse_rejected_betas ──────────────────────────────────────────────────

    #[test]
    fn parse_empty_message_returns_empty() {
        let result = parse_rejected_betas("", &candidates());
        assert!(result.is_empty());
    }

    #[test]
    fn parse_no_match_returns_empty() {
        let result = parse_rejected_betas(
            "The request is invalid for some unrelated reason.",
            &candidates(),
        );
        assert!(result.is_empty());
    }

    #[test]
    fn parse_single_beta_named_in_error() {
        let msg =
            "ValidationException: The beta feature 'claude-code-x' is not supported on this model.";
        let result = parse_rejected_betas(msg, &candidates());
        assert_eq!(result, vec!["claude-code-x".to_string()]);
    }

    #[test]
    fn parse_multiple_betas_named_in_error() {
        let msg = "ValidationException: features context-1m-2025-08-07 and claude-code-x are not supported.";
        let mut result = parse_rejected_betas(msg, &candidates());
        result.sort();
        let mut expected = vec![
            "context-1m-2025-08-07".to_string(),
            "claude-code-x".to_string(),
        ];
        expected.sort();
        assert_eq!(result, expected);
    }

    #[test]
    fn parse_case_insensitive_match() {
        // Bedrock might capitalise or alter casing in error messages.
        let msg = "Unsupported beta: CONTEXT-1M-2025-08-07";
        let result = parse_rejected_betas(msg, &candidates());
        assert_eq!(result, vec!["context-1m-2025-08-07".to_string()]);
    }

    #[test]
    fn parse_empty_candidates_returns_empty() {
        let result = parse_rejected_betas("context-1m-2025-08-07 is not supported", &[]);
        assert!(result.is_empty());
    }

    #[test]
    fn parse_does_not_duplicate_when_beta_appears_twice_in_message() {
        // The beta appears twice but candidates only contains it once — result should have one entry.
        let msg = "context-1m-2025-08-07 context-1m-2025-08-07 rejected";
        let result = parse_rejected_betas(msg, &candidates());
        assert_eq!(result, vec!["context-1m-2025-08-07".to_string()]);
    }
}
