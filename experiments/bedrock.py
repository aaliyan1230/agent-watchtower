"""Minimal Amazon Bedrock InvokeModel client (stdlib + httpx only).

Mirror of the Go reference in watchtower/internal/bedrock: SigV4-signed
POST /model/{model}/invoke for third-party (OpenAI-shape) models such as
deepseek.v3.2 and moonshotai.kimi-k2.5. The judge needs one JSON round
trip per prompt, and implementing SigV4 by hand — canonical request ->
string to sign -> HMAC key chain — keeps the harness dependency-light
and makes the signing auditable (the Go side already teaches the same
algorithm).

Credentials come from the environment and are resolved exactly like the
experiment live runner resolves them for the Go server (experiments.run.
aws_env): .env wins, otherwise the AWS CLI session is forwarded.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass
from typing import Mapping

import httpx
from urllib.parse import quote

# Service name for SigV4 credential scope (matches the Go client).
SERVICE = "bedrock"


@dataclass
class Credentials:
    access_key: str
    secret_key: str
    session_token: str = ""


def credentials_from_env(env: Mapping[str, str] | None = None) -> Credentials:
    """Read AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_SESSION_TOKEN
    from the environment. Missing keys fail loudly — an unsigned judge
    call would only fail with a confusing auth error later."""
    env = env if env is not None else os.environ
    creds = Credentials(
        access_key=env.get("AWS_ACCESS_KEY_ID", ""),
        secret_key=env.get("AWS_SECRET_ACCESS_KEY", ""),
        session_token=env.get("AWS_SESSION_TOKEN", ""),
    )
    if not creds.access_key or not creds.secret_key:
        raise ValueError(
            "bedrock judge needs AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY "
            "in .env or the AWS CLI; missing keys fail loudly"
        )
    return creds


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hmac_sha256(key: bytes, data: str) -> bytes:
    return hmac.new(key, data.encode("utf-8"), hashlib.sha256).digest()


def _now_amz() -> tuple[str, str]:
    """(amz-date, date-stamp) for SigV4 — module-level so tests can pin it."""
    now = time.gmtime()
    amz = time.strftime("%Y%m%dT%H%M%SZ", now)
    return amz, amz[:8]


class Client:
    """SigV4-signed Bedrock InvokeModel client for one model in one region.

    Only the third-party OpenAI-shape payload is needed by the CausalTrace
    judge (deepseek / moonshot / qwen), so only that shape is implemented;
    `amazon.*` native models are out of scope for this experiment.
    """

    def __init__(
        self,
        model: str,
        region: str,
        creds: Credentials,
        base_url: str | None = None,
        http: httpx.Client | None = None,
    ):
        if not creds.access_key or not creds.secret_key:
            raise ValueError("bedrock: missing AWS credentials")
        self.model = model
        self.region = region
        self.creds = creds
        self._base = (
            base_url or f"https://bedrock-runtime.{region}.amazonaws.com"
        ).rstrip("/")
        self._client = http or httpx.Client(timeout=60.0)

    # -- signing -----------------------------------------------------------

    def _signed_headers(
        self, body: bytes, amz_date: str, date_stamp: str
    ) -> dict[str, str]:
        payload_hash = _sha256_hex(body)
        host = self._base.split("://", 1)[-1]
        headers: dict[str, str] = {
            "content-type": "application/json",
            "host": host,
            "x-amz-date": amz_date,
        }
        if self.creds.session_token:
            headers["x-amz-security-token"] = self.creds.session_token
        signed = sorted(headers)

        canonical_headers = "".join(f"{k}:{headers[k]}\n" for k in signed)
        # AWS canonicalizes the received path by re-encoding it, so the
        # signed canonical URI percent-encodes the model id (":" -> %3A);
        # boto3 sends the encoded path on the wire too, and AWS accepts it.
        canonical_uri = "/model/{}/invoke".format(quote(self.model, safe=""))
        canonical_request = "\n".join(
            [
                "POST",
                canonical_uri,
                "",
                canonical_headers,
                ";".join(signed),
                payload_hash,
            ]
        )

        scope = f"{date_stamp}/{self.region}/{SERVICE}/aws4_request"
        string_to_sign = "\n".join(
            [
                "AWS4-HMAC-SHA256",
                amz_date,
                scope,
                _sha256_hex(canonical_request.encode("utf-8")),
            ]
        )

        k_date = _hmac_sha256(b"AWS4" + self.creds.secret_key.encode(), date_stamp)
        k_region = _hmac_sha256(k_date, self.region)
        k_service = _hmac_sha256(k_region, SERVICE)
        k_signing = _hmac_sha256(k_service, "aws4_request")
        signature = hmac.new(
            k_signing, string_to_sign.encode("utf-8"), hashlib.sha256
        ).hexdigest()

        headers["Authorization"] = (
            f"AWS4-HMAC-SHA256 Credential={self.creds.access_key}/{scope}, "
            f"SignedHeaders={';'.join(signed)}, Signature={signature}"
        )
        return headers

    # -- calls -------------------------------------------------------------

    def chat(self, system: str, user: str) -> tuple[str, dict]:
        """One system+user turn; returns (text, usage) for the judge. usage
        carries prompt/completion tokens — judge cost is measured, not
        guessed."""
        body = json.dumps(
            {
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "max_tokens": 2048,
                "temperature": 0,
            },
            separators=(",", ":"),
        ).encode("utf-8")
        amz_date, date_stamp = _now_amz()
        headers = self._signed_headers(body, amz_date, date_stamp)
        path = "/model/{}/invoke".format(quote(self.model, safe=""))
        resp = self._client.post(f"{self._base}{path}", headers=headers, content=body)
        if resp.status_code != 200:
            raise ValueError(f"bedrock: {resp.status_code}: {resp.text[:300]}")
        return self._parse(resp.json())

    def _parse(self, data: Mapping) -> tuple[str, dict]:
        choices = data.get("choices") or []
        if not choices or not choices[0].get("message", {}).get("content"):
            raise ValueError(f"bedrock: empty response from {self.model}")
        usage = data.get("usage") or {}
        return (
            choices[0]["message"]["content"],
            {
                "inputTokens": int(usage.get("prompt_tokens", 0)),
                "outputTokens": int(usage.get("completion_tokens", 0)),
            },
        )
