"""Python Bedrock SigV4 transport: signing primitive vs AWS vectors,
header structure, and wire parsing."""

import json

import httpx
import pytest

from experiments import bedrock


def test_hmac_key_chain_matches_aws_vectors():
    # AWS Signature Version 4 test suite, key derivation example:
    # key "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY", datestamp 20120215.
    # kDate below is the value AWS publishes in its docs; the rest were
    # derived from it and cross-checked against a plain hashlib chain, so
    # these goldens lock the KDF against accidental changes.
    key = b"AWS4wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY"
    k_date = bedrock._hmac_sha256(key, "20120215")
    assert (
        k_date.hex()
        == "969fbb94feb542b71ede6f87fe4d5fa29c789342b0f407474670f0c2489e0a0d"
    )
    k_region = bedrock._hmac_sha256(k_date, "us-east-1")
    assert (
        k_region.hex()
        == "69daa0209cd9c5ff5c8ced464a696fd4252e981430b10e3d3fd8e2f197d7a70c"
    )
    k_service = bedrock._hmac_sha256(k_region, "iam")
    assert (
        k_service.hex()
        == "f72cfd46f26bc4643f06a11eabb6c0ba18780c19a8da0c31ace671265e3c87fa"
    )
    k_signing = bedrock._hmac_sha256(k_service, "aws4_request")
    assert (
        k_signing.hex()
        == "f4780e2d9f65fa895f9c67b32ce1baf0b0d8a43505a000a1a9e090d414db404d"
    )


def test_signed_headers_structure_and_stability(monkeypatch):
    monkeypatch.setattr(bedrock, "_now_amz", lambda: ("20260101T000000Z", "20260101"))
    creds = bedrock.Credentials(
        access_key="AKIAEXAMPLE", secret_key="SECRET", session_token="TOKEN"
    )
    client = bedrock.Client("deepseek.v3.2", "us-east-1", creds)
    body = b'{"messages":[{"role":"system","content":"s"}],"max_tokens":2048}'
    headers = client._signed_headers(body, "20260101T000000Z", "20260101")

    assert headers["content-type"] == "application/json"
    assert headers["x-amz-date"] == "20260101T000000Z"
    assert headers["x-amz-security-token"] == "TOKEN"
    auth = headers["Authorization"]
    assert auth.startswith(
        "AWS4-HMAC-SHA256 Credential=AKIAEXAMPLE/20260101/us-east-1/bedrock/aws4_request"
    )
    assert "SignedHeaders=content-type;host;x-amz-date;x-amz-security-token" in auth
    assert auth.count("Signature=") == 1

    # Deterministic: identical inputs -> identical signature.
    again = client._signed_headers(body, "20260101T000000Z", "20260101")
    assert again["Authorization"] == auth
    # A different date changes the scope and signature.
    other = client._signed_headers(body, "20260102T000000Z", "20260102")
    assert other["Authorization"] != auth


def test_chat_post_canonical_uri_and_parse():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization", "")
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "PASS"}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 3},
            },
        )

    client = bedrock.Client(
        "moonshotai.kimi-k2.5",
        "us-east-1",
        bedrock.Credentials("AK", "SK"),
        base_url="https://bedrock-runtime.us-east-1.amazonaws.com",
        http=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    text, usage = client.chat("system", "user")

    assert text == "PASS"
    assert usage == {"inputTokens": 12, "outputTokens": 3}
    # Wire path targets the model's invoke endpoint.
    assert seen["url"].startswith(
        "https://bedrock-runtime.us-east-1.amazonaws.com/model/"
    )
    assert seen["url"].endswith("/invoke")
    assert "kimi-k2.5" in seen["url"], seen["url"]
    assert seen["auth"].startswith("AWS4-HMAC-SHA256 ")
    assert seen["body"]["messages"][0]["role"] == "system"
    assert seen["body"]["temperature"] == 0


def test_colon_in_model_id_is_percent_encoded():
    # A model id containing ':' must go out as "%3A" on the wire (AWS
    # re-canonicalizes the received path against the %3A form).
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"choices": [{"message": {"content": "PASS"}}]})

    client = bedrock.Client(
        "acme:model-x",
        "us-east-1",
        bedrock.Credentials("AK", "SK"),
        http=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    client.chat("system", "user")
    assert "/model/acme%3Amodel-x/invoke" in seen["url"], seen["url"]


def test_chat_http_error_raises():
    client = bedrock.Client(
        "deepseek.v3.2",
        "us-east-1",
        bedrock.Credentials("AK", "SK"),
        http=httpx.Client(
            transport=httpx.MockTransport(
                lambda r: httpx.Response(403, text="AccessDenied")
            )
        ),
    )
    with pytest.raises(ValueError, match="403"):
        client.chat("system", "user")


def test_chat_empty_response_raises():
    client = bedrock.Client(
        "deepseek.v3.2",
        "us-east-1",
        bedrock.Credentials("AK", "SK"),
        http=httpx.Client(
            transport=httpx.MockTransport(
                lambda r: httpx.Response(200, json={"choices": []})
            )
        ),
    )
    with pytest.raises(ValueError, match="empty response"):
        client.chat("system", "user")


def test_credentials_from_env_requires_keys(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AK")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "")
    with pytest.raises(ValueError, match="AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY"):
        bedrock.credentials_from_env()
    c = bedrock.credentials_from_env(
        {"AWS_ACCESS_KEY_ID": "AK", "AWS_SECRET_ACCESS_KEY": "SK"}
    )
    assert c == bedrock.Credentials(access_key="AK", secret_key="SK")
