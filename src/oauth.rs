//! OAuth-token source for usage snapshots.
//!
//! Reads the OAuth credentials Claude Code stashes in the macOS Keychain under
//! service `Claude Code-credentials` and queries the OAuth-authenticated
//! endpoints on `api.anthropic.com`. This is the sole usage source: it avoids
//! the browser-cookie / Cloudflare path entirely and covers the account the
//! active Claude Code CLI is logged into.
//!
//! Endpoints used (token scope `user:profile` is sufficient):
//!   GET https://api.anthropic.com/api/oauth/usage     (rolling-window quotas + extra_usage)
//!   GET https://api.anthropic.com/api/oauth/profile   (account email + org uuid)
//!
//! Token refresh: when Claude Code's CLI is running it rotates the access
//! token itself (about 5 minutes before expiry), and we just read whatever is
//! in the keychain. But many users only use the Claude desktop app, which keeps
//! its own login, so nothing rotates the CLI's token and the menu bar would be
//! stuck on "token expired" until they happen to run `claude`. So once the
//! token has actually expired (or is within `REFRESH_SKEW_MS` of it, a window
//! the running CLI would already have refreshed in), we refresh it ourselves
//! with the stored refresh token, exactly the way the CLI does, and write the
//! rotated credentials back to the same keychain item so the CLI stays
//! logged in. See `refresh_creds`.

use anyhow::{Context, Result};
use chrono::Utc;
use rquest::Client;
use serde::Deserialize;
use std::time::Duration;

use crate::models::{UsageResponse, UsageSnapshot};

const KEYCHAIN_SERVICE: &str = "Claude Code-credentials";
const API_BASE: &str = "https://api.anthropic.com";
/// Token endpoint and public OAuth client ID used by the Claude Code CLI
/// (`TOKEN_URL` / `CLIENT_ID` in its bundled config).
const TOKEN_URL: &str = "https://platform.claude.com/v1/oauth/token";
const CLAUDE_CODE_CLIENT_ID: &str = "9d1c250a-e61b-44d9-88ed-5944d1962f5e";
/// Refresh when the token expires within this window. Kept well under the
/// CLI's own ~5 minute pre-expiry refresh so we never race a running CLI.
const REFRESH_SKEW_MS: i64 = 60_000;
/// `security -i` reads commands line by line; very long lines get truncated.
/// Matches the CLI's behaviour of falling back to argv for oversized blobs.
const SECURITY_STDIN_LINE_LIMIT: usize = 4000;
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

/// What the keychain blob looks like:
/// ```json
/// {"claudeAiOauth": {
///   "accessToken": "sk-ant-oat01-...",
///   "refreshToken": "sk-ant-ort01-...",
///   "expiresAt": 1778299177154,
///   "scopes": ["user:profile", "user:inference", ...],
///   "subscriptionType": "max",
///   "rateLimitTier": "default_claude_max_20x"
/// }}
/// ```
#[derive(Deserialize)]
struct KeychainBlob {
    #[serde(rename = "claudeAiOauth")]
    oauth: OAuthCreds,
}

#[derive(Deserialize)]
pub struct OAuthCreds {
    #[serde(rename = "accessToken")]
    pub access_token: String,
    #[serde(rename = "refreshToken", default)]
    pub refresh_token: Option<String>,
    #[serde(rename = "expiresAt")]
    pub expires_at: i64,
    #[serde(default)]
    pub scopes: Vec<String>,
    #[serde(rename = "subscriptionType", default)]
    pub subscription_type: Option<String>,
    #[serde(rename = "rateLimitTier", default)]
    pub rate_limit_tier: Option<String>,
}

pub fn read_token() -> Result<OAuthCreds> {
    parse_creds(&read_keychain_raw(KEYCHAIN_SERVICE)?)
}

/// Raw JSON blob stored in the keychain item.
fn read_keychain_raw(service: &str) -> Result<String> {
    let out = std::process::Command::new("/usr/bin/security")
        .args(["find-generic-password", "-s", service, "-w"])
        .output()
        .context("spawn /usr/bin/security")?;
    if !out.status.success() {
        anyhow::bail!(
            "`security find-generic-password -s \"{service}\"` failed: {}. Is Claude Code logged in?",
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    String::from_utf8(out.stdout).context("Claude Code keychain blob was not valid UTF-8")
}

fn parse_creds(raw: &str) -> Result<OAuthCreds> {
    let trimmed = raw.trim();
    let blob: KeychainBlob = serde_json::from_str(trimmed).map_err(|e| {
        // Common case: the keychain item exists and parses, but has no
        // `claudeAiOauth` block — only `mcpOAuth` plugin tokens. That means
        // Claude Code is installed but the user never logged in with a Claude
        // Pro/Max subscription (they're on an API key, or only configured MCP
        // plugins). Surface a plain-English message instead of raw serde noise.
        if e.to_string().contains("missing field `claudeAiOauth`")
            && trimmed.contains("\"mcpOAuth\"")
        {
            anyhow::anyhow!(
                "Claude Code is installed but not logged into a Claude Pro/Max subscription. \
                 Run `claude` in Terminal and choose \"Log in with your Claude account\" \
                 (not an API key), then click Refresh now."
            )
        } else {
            let snippet = &raw[..raw.len().min(80)];
            anyhow::Error::new(e)
                .context(format!("parse Claude Code credentials JSON (head: {snippet:?})"))
        }
    })?;
    Ok(blob.oauth)
}

#[derive(Deserialize)]
struct OAuthProfile {
    account: ProfileAccount,
    organization: ProfileOrg,
}

#[derive(Deserialize)]
struct ProfileAccount {
    email: Option<String>,
}

#[derive(Deserialize)]
struct ProfileOrg {
    uuid: String,
}

pub async fn fetch_oauth_snapshot() -> Result<UsageSnapshot> {
    let mut creds = read_token().context("read OAuth token from Keychain")?;

    // Plain rquest client; api.anthropic.com is not behind Cloudflare's bot
    // gate, so no Chrome fingerprint emulation needed (unlike claude.ai).
    let client = Client::builder()
        .connect_timeout(CONNECT_TIMEOUT)
        .read_timeout(REQUEST_TIMEOUT)
        .timeout(REQUEST_TIMEOUT)
        .build()
        .context("build rquest client")?;

    if needs_refresh(&creds, Utc::now().timestamp_millis()) {
        creds = refresh_creds(&client, creds, KEYCHAIN_SERVICE, TOKEN_URL).await?;
    }
    let token = creds.access_token.as_str();

    // /api/oauth/usage carries the rolling-window utilization AND the
    // extra_usage block, so we don't need a separate overage_spend_limit call.
    let usage: UsageResponse = get_json(&client, token, &format!("{API_BASE}/api/oauth/usage"))
        .await
        .context("fetch /api/oauth/usage")?;

    // /api/oauth/profile gives us account email + org uuid for display.
    let profile: OAuthProfile = get_json(&client, token, &format!("{API_BASE}/api/oauth/profile"))
        .await
        .context("fetch /api/oauth/profile")?;

    Ok(UsageSnapshot {
        org_uuid: profile.organization.uuid,
        browser: "Claude Code".to_string(),
        account_email: profile.account.email,
        fetched_at: Utc::now(),
        usage: Some(usage),
        // Stripe-flavoured subscription_details and the dedicated
        // overage_spend_limit endpoint are not exposed via OAuth scopes, so
        // these fields stay empty.
        overage: None,
        subscription: None,
        errors: Vec::new(),
        stale: false,
    })
}

fn needs_refresh(creds: &OAuthCreds, now_ms: i64) -> bool {
    creds.expires_at > 0 && creds.expires_at - REFRESH_SKEW_MS < now_ms
}

#[derive(Deserialize)]
struct TokenResponse {
    access_token: String,
    #[serde(default)]
    refresh_token: Option<String>,
    expires_in: i64,
    #[serde(default)]
    scope: Option<String>,
}

/// Exchange the stored refresh token for a new access token and persist the
/// rotated credentials to the keychain, the same way the Claude Code CLI does.
///
/// Ordering matters: refresh tokens rotate, so once the token endpoint
/// answers, the refresh token the CLI has stored is dead. If we then failed
/// to write the new one back, the user would be logged out of Claude Code.
/// So we first prove we can write the keychain item (by writing back the
/// exact blob that's already there) and only then spend the refresh token.
async fn refresh_creds(
    client: &Client,
    creds: OAuthCreds,
    service: &str,
    token_url: &str,
) -> Result<OAuthCreds> {
    let expired_msg = |why: String| {
        let age_min = (Utc::now().timestamp_millis() - creds.expires_at) / 60_000;
        anyhow::anyhow!(
            "Claude Code OAuth token expired {age_min} minutes ago and could not be refreshed ({why}). \
             Run `claude` in Terminal once to log in again, then click Refresh now."
        )
    };
    let Some(refresh_token) = creds.refresh_token.clone() else {
        return Err(expired_msg("no refresh token in keychain".into()));
    };

    let raw = read_keychain_raw(service)?;
    let account = keychain_account(service)?;
    write_keychain_raw(service, &account, raw.trim())
        .map_err(|e| expired_msg(format!("keychain not writable: {e:#}")))?;

    let mut body = serde_json::json!({
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": CLAUDE_CODE_CLIENT_ID,
    });
    if !creds.scopes.is_empty() {
        body["scope"] = serde_json::Value::String(creds.scopes.join(" "));
    }
    let resp = client
        .post(token_url)
        .header("Content-Type", "application/json")
        .header("Accept", "application/json")
        .body(body.to_string())
        .send()
        .await
        .with_context(|| format!("POST {token_url}"))?;
    let status = resp.status();
    let text = resp.text().await.unwrap_or_default();
    if !status.is_success() {
        // A running CLI may have rotated the token between our read and our
        // POST, which spends the refresh token we sent. If the keychain now
        // holds a fresh token, just use that.
        if let Ok(now) = read_keychain_raw(service).and_then(|r| parse_creds(&r)) {
            if now.access_token != creds.access_token
                && !needs_refresh(&now, Utc::now().timestamp_millis())
            {
                return Ok(now);
            }
        }
        let snippet = &text[..text.len().min(200)];
        return Err(expired_msg(format!("HTTP {status} from token endpoint: {snippet}")));
    }
    let tok: TokenResponse =
        serde_json::from_str(&text).context("parse OAuth token refresh response")?;

    // Merge into the blob as it is right now (not the copy from before the
    // POST) so we never clobber unrelated fields such as `mcpOAuth`.
    let current = read_keychain_raw(service).unwrap_or(raw);
    let mut blob: serde_json::Value =
        serde_json::from_str(current.trim()).context("parse Claude Code credentials JSON")?;
    let oauth = blob
        .get_mut("claudeAiOauth")
        .and_then(|v| v.as_object_mut())
        .context("claudeAiOauth missing from keychain blob")?;
    let new_refresh = tok.refresh_token.unwrap_or(refresh_token);
    let new_expires = Utc::now().timestamp_millis() + tok.expires_in * 1000;
    oauth.insert("accessToken".into(), tok.access_token.clone().into());
    oauth.insert("refreshToken".into(), new_refresh.clone().into());
    oauth.insert("expiresAt".into(), new_expires.into());
    let scopes = match tok.scope {
        Some(s) if !s.trim().is_empty() => s.split_whitespace().map(str::to_string).collect(),
        _ => creds.scopes,
    };
    oauth.insert("scopes".into(), serde_json::json!(scopes));
    write_keychain_raw(service, &account, &blob.to_string()).context(
        "token was refreshed but writing it back to the keychain failed; run `claude` in Terminal to log in again",
    )?;

    Ok(OAuthCreds {
        access_token: tok.access_token,
        refresh_token: Some(new_refresh),
        expires_at: new_expires,
        scopes,
        subscription_type: creds.subscription_type,
        rate_limit_tier: creds.rate_limit_tier,
    })
}

/// Account attribute of the existing keychain item, so `add-generic-password
/// -U` updates that item instead of creating a second one next to it.
fn keychain_account(service: &str) -> Result<String> {
    let out = std::process::Command::new("/usr/bin/security")
        .args(["find-generic-password", "-s", service])
        .output()
        .context("spawn /usr/bin/security")?;
    if !out.status.success() {
        anyhow::bail!(
            "`security find-generic-password -s \"{service}\"` failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    let attrs = String::from_utf8_lossy(&out.stdout);
    attrs
        .lines()
        .find_map(|l| {
            l.trim()
                .strip_prefix("\"acct\"<blob>=\"")
                .and_then(|r| r.strip_suffix('"'))
                .map(str::to_string)
        })
        .context("could not read account name of the Claude Code keychain item")
}

/// Write `json` as the keychain item's secret. Mirrors the CLI: hex-encode the
/// payload and feed the command to `security -i` on stdin so the token never
/// shows up in the process list; fall back to argv only for huge blobs.
fn write_keychain_raw(service: &str, account: &str, json: &str) -> Result<()> {
    use std::io::Write;
    use std::process::{Command, Stdio};

    if [account, service].iter().any(|s| s.contains('"') || s.contains('\n')) {
        anyhow::bail!("unexpected characters in keychain account name");
    }
    let hex: String = json.bytes().map(|b| format!("{b:02x}")).collect();
    let line = format!(
        "add-generic-password -U -a \"{account}\" -s \"{service}\" -X \"{hex}\"\n"
    );
    let out = if line.len() <= SECURITY_STDIN_LINE_LIMIT {
        let mut child = Command::new("/usr/bin/security")
            .arg("-i")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .context("spawn /usr/bin/security -i")?;
        child
            .stdin
            .take()
            .context("open security stdin")?
            .write_all(line.as_bytes())
            .context("write to security stdin")?;
        child.wait_with_output().context("wait for security")?
    } else {
        Command::new("/usr/bin/security")
            .args([
                "add-generic-password",
                "-U",
                "-a",
                account,
                "-s",
                service,
                "-X",
                &hex,
            ])
            .output()
            .context("spawn /usr/bin/security")?
    };
    // `security -i` exits 0 even when a command fails, so check stderr too.
    let stderr = String::from_utf8_lossy(&out.stderr);
    if !out.status.success() || !stderr.trim().is_empty() {
        anyhow::bail!("security add-generic-password failed: {}", stderr.trim());
    }
    // Read back to confirm the item now holds exactly what we wrote.
    let back = read_keychain_raw(service)?;
    if back.trim() != json {
        anyhow::bail!("keychain write did not stick (read-back mismatch)");
    }
    Ok(())
}

async fn get_json<T: serde::de::DeserializeOwned>(
    client: &Client,
    token: &str,
    url: &str,
) -> Result<T> {
    let resp = client
        .get(url)
        .header("Authorization", format!("Bearer {token}"))
        .header("Accept", "application/json")
        .header("anthropic-version", "2023-06-01")
        .send()
        .await
        .with_context(|| format!("GET {url}"))?;
    let status = resp.status();
    // Capture Retry-After before we move the response. Only meaningful on
    // 429; other statuses sometimes echo a stale header value. RFC 7231 says
    // it's either delta-seconds or an HTTP-date — Anthropic returns seconds.
    // We surface it through the error message so the menubar's poll loop
    // can sleep for the server-requested duration instead of guessing.
    let retry_after = if status.as_u16() == 429 {
        resp.headers()
            .get("retry-after")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.trim().parse::<u64>().ok())
    } else {
        None
    };
    let body = resp.text().await.unwrap_or_default();
    if !status.is_success() {
        let snippet = &body[..body.len().min(300)];
        match retry_after {
            Some(secs) => {
                anyhow::bail!("HTTP {status} from {url} (retry-after={secs}s): {snippet}")
            }
            None => anyhow::bail!("HTTP {status} from {url}: {snippet}"),
        }
    }
    serde_json::from_str::<T>(&body).with_context(|| {
        let snippet = &body[..body.len().min(300)];
        format!("parse JSON from {url}: {snippet}")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn creds(expires_at: i64) -> OAuthCreds {
        OAuthCreds {
            access_token: "a".into(),
            refresh_token: Some("r".into()),
            expires_at,
            scopes: vec![],
            subscription_type: None,
            rate_limit_tier: None,
        }
    }

    #[test]
    fn refresh_only_when_expired_or_about_to() {
        let now = 1_000_000_000;
        assert!(needs_refresh(&creds(now - 1), now));
        assert!(needs_refresh(&creds(now + REFRESH_SKEW_MS - 1), now));
        assert!(!needs_refresh(&creds(now + REFRESH_SKEW_MS + 1), now));
        // expiresAt == 0 means "no expiry recorded"; leave it alone.
        assert!(!needs_refresh(&creds(0), now));
    }

    #[test]
    fn parses_refresh_token_and_scopes() {
        let c = parse_creds(
            r#"{"claudeAiOauth":{"accessToken":"x","refreshToken":"y","expiresAt":5,"scopes":["user:profile","user:inference"]}}"#,
        )
        .unwrap();
        assert_eq!(c.refresh_token.as_deref(), Some("y"));
        assert_eq!(c.scopes, vec!["user:profile", "user:inference"]);
    }

    /// End-to-end refresh against a throwaway keychain item and a local mock
    /// token endpoint. Touches the login keychain, so it is opt-in:
    /// `cargo test --lib -- --ignored refresh_round_trip`
    #[tokio::test]
    #[ignore]
    async fn refresh_round_trip() {
        use std::io::{Read, Write};
        let service = "claude-meter-test-credentials";
        let _ = std::process::Command::new("/usr/bin/security")
            .args(["delete-generic-password", "-s", service])
            .output();
        let original = r#"{"claudeAiOauth":{"accessToken":"old-access","refreshToken":"old-refresh","expiresAt":1000,"scopes":["user:profile","user:inference"],"subscriptionType":"pro"},"mcpOAuth":{"keep":"me"}}"#;
        let st = std::process::Command::new("/usr/bin/security")
            .args(["add-generic-password", "-a", "tester", "-s", service, "-w", original])
            .status()
            .unwrap();
        assert!(st.success());

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/v1/oauth/token", listener.local_addr().unwrap());
        let server = std::thread::spawn(move || {
            let (mut sock, _) = listener.accept().unwrap();
            let mut buf = vec![0u8; 8192];
            let mut req = Vec::new();
            loop {
                let n = sock.read(&mut buf).unwrap();
                req.extend_from_slice(&buf[..n]);
                let text = String::from_utf8_lossy(&req);
                if let Some(i) = text.find("\r\n\r\n") {
                    let len = text[..i]
                        .lines()
                        .find_map(|l| l.to_ascii_lowercase().strip_prefix("content-length:").map(|v| v.trim().parse::<usize>().unwrap()))
                        .unwrap_or(0);
                    if req.len() >= i + 4 + len {
                        break;
                    }
                }
                if n == 0 {
                    break;
                }
            }
            let body = r#"{"access_token":"new-access","refresh_token":"new-refresh","expires_in":28800,"scope":"user:profile user:inference"}"#;
            write!(sock, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", body.len(), body).unwrap();
            String::from_utf8(req).unwrap()
        });

        let client = Client::builder().build().unwrap();
        let creds = parse_creds(original).unwrap();
        let result = refresh_creds(&client, creds, service, &url).await;
        let request = server.join().unwrap();
        let stored = read_keychain_raw(service);
        let _ = std::process::Command::new("/usr/bin/security")
            .args(["delete-generic-password", "-s", service])
            .output();

        let new = result.unwrap();
        assert_eq!(new.access_token, "new-access");
        assert_eq!(new.refresh_token.as_deref(), Some("new-refresh"));
        assert!(new.expires_at > Utc::now().timestamp_millis() + 7 * 3600 * 1000);

        let req_body: serde_json::Value =
            serde_json::from_str(request.split("\r\n\r\n").nth(1).unwrap()).unwrap();
        assert_eq!(req_body["grant_type"], "refresh_token");
        assert_eq!(req_body["refresh_token"], "old-refresh");
        assert_eq!(req_body["client_id"], CLAUDE_CODE_CLIENT_ID);
        assert_eq!(req_body["scope"], "user:profile user:inference");

        let stored: serde_json::Value = serde_json::from_str(stored.unwrap().trim()).unwrap();
        assert_eq!(stored["claudeAiOauth"]["accessToken"], "new-access");
        assert_eq!(stored["claudeAiOauth"]["refreshToken"], "new-refresh");
        assert_eq!(stored["claudeAiOauth"]["subscriptionType"], "pro");
        assert_eq!(stored["mcpOAuth"]["keep"], "me");
    }
}
