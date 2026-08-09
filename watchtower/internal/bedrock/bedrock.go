// Package bedrock is a minimal Amazon Bedrock client for the judge's
// transport: SigV4-signed InvokeModel calls, stdlib-only. The judge
// needs one endpoint and one JSON round trip, and implementing SigV4
// by hand keeps the module lean — and teaches the signing algorithm
// (canonical request -> string to sign -> HMAC key chain).
package bedrock

import (
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"sort"
	"strings"
	"time"
)

// Credentials are the AWS long-term or session credentials the
// Bedrock judge signs with. Keys come from the environment only.
type Credentials struct {
	AccessKey    string
	SecretKey    string
	SessionToken string
}

// CredentialsFromEnv reads AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY /
// AWS_SESSION_TOKEN (optional). Missing keys fail loudly — an unsigned
// judge call would return a confusing auth error later.
func CredentialsFromEnv() (Credentials, error) {
	c := Credentials{
		AccessKey:    os.Getenv("AWS_ACCESS_KEY_ID"),
		SecretKey:    os.Getenv("AWS_SECRET_ACCESS_KEY"),
		SessionToken: os.Getenv("AWS_SESSION_TOKEN"),
	}
	if c.AccessKey == "" || c.SecretKey == "" {
		return c, fmt.Errorf("bedrock: AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set in the environment")
	}
	return c, nil
}

// Client signs and sends InvokeModel requests for one model in one
// region.
type Client struct {
	model   string
	region  string
	creds   Credentials
	http    *http.Client
	baseURL string
}

func New(model, region string, creds Credentials) *Client {
	return &Client{
		model:   model,
		region:  region,
		creds:   creds,
		http:    &http.Client{Timeout: 60 * time.Second},
		baseURL: fmt.Sprintf("https://bedrock-runtime.%s.amazonaws.com", region),
	}
}

// ChatText sends a system+user pair to the model and returns its text
// response. Two native shapes exist: Amazon's own models (amazon.*)
// take a top-level `system` array with text blocks and an
// inferenceConfig; third-party models (deepseek, moonshot, qwen, ...)
// take the OpenAI shape — system as a message role, plain string
// content, max_tokens at top level.
func (c *Client) ChatText(ctx context.Context, system, user string) (string, error) {
	body, err := json.Marshal(c.requestBody(system, user))
	if err != nil {
		return "", err
	}
	raw, err := c.invoke(ctx, body)
	if err != nil {
		return "", err
	}
	return c.parseResponse(raw)
}

func (c *Client) requestBody(system, user string) map[string]any {
	if strings.HasPrefix(c.model, "amazon.") {
		return map[string]any{
			"system":          []map[string]any{{"text": system}},
			"messages":        []map[string]any{{"role": "user", "content": []map[string]any{{"text": user}}}},
			"inferenceConfig": map[string]any{"max_new_tokens": 2048, "temperature": 0},
		}
	}
	return map[string]any{
		"messages": []map[string]any{
			{"role": "system", "content": system},
			{"role": "user", "content": user},
		},
		"max_tokens":  2048,
		"temperature": 0,
	}
}

func (c *Client) parseResponse(raw []byte) (string, error) {
	if strings.HasPrefix(c.model, "amazon.") {
		var decoded struct {
			Output struct {
				Message struct {
					Content []struct {
						Text string `json:"text"`
					} `json:"content"`
				} `json:"message"`
			} `json:"output"`
		}
		if err := json.Unmarshal(raw, &decoded); err != nil {
			return "", fmt.Errorf("bedrock: decode response: %w", err)
		}
		text := ""
		for _, block := range decoded.Output.Message.Content {
			text += block.Text
		}
		if text == "" {
			return "", fmt.Errorf("bedrock: empty response from %s", c.model)
		}
		return text, nil
	}
	var decoded struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(raw, &decoded); err != nil {
		return "", fmt.Errorf("bedrock: decode response: %w", err)
	}
	if len(decoded.Choices) == 0 || decoded.Choices[0].Message.Content == "" {
		return "", fmt.Errorf("bedrock: empty response from %s", c.model)
	}
	return decoded.Choices[0].Message.Content, nil
}

func (c *Client) invoke(ctx context.Context, body []byte) ([]byte, error) {
	host := strings.TrimPrefix(c.baseURL, "https://")
	// AWS canonicalizes the received path by re-encoding it, so the
	// signed canonical URI percent-encodes the model id (":" -> %3A)
	// while the wire path keeps the literal colon. Go's EscapedPath
	// would re-encode the colon too, so Opaque carries the raw path.
	rawPath := "/model/" + c.model + "/invoke"
	canonicalURI := "/model/" + url.QueryEscape(c.model) + "/invoke"
	amzDate := time.Now().UTC().Format("20060102T150405Z")
	dateStamp := amzDate[:8]

	payloadHash := sha256Hex(body)
	headers := map[string]string{
		"content-type": "application/json",
		"host":         host,
		"x-amz-date":   amzDate,
	}
	if c.creds.SessionToken != "" {
		headers["x-amz-security-token"] = c.creds.SessionToken
	}
	signedHeaders := sortedKeys(headers)

	canonicalHeaders := ""
	for _, k := range signedHeaders {
		canonicalHeaders += k + ":" + strings.TrimSpace(headers[k]) + "\n"
	}
	canonicalRequest := strings.Join([]string{
		http.MethodPost, canonicalURI, "", canonicalHeaders, strings.Join(signedHeaders, ";"), payloadHash,
	}, "\n")

	scope := dateStamp + "/" + c.region + "/bedrock/aws4_request"
	stringToSign := strings.Join([]string{
		"AWS4-HMAC-SHA256", amzDate, scope, sha256Hex([]byte(canonicalRequest)),
	}, "\n")

	kDate := hmacSHA256([]byte("AWS4"+c.creds.SecretKey), dateStamp)
	kRegion := hmacSHA256(kDate, c.region)
	kService := hmacSHA256(kRegion, "bedrock")
	kSigning := hmacSHA256(kService, "aws4_request")
	signature := hex.EncodeToString(hmacSHA256(kSigning, stringToSign))

	authorization := fmt.Sprintf(
		"AWS4-HMAC-SHA256 Credential=%s/%s, SignedHeaders=%s, Signature=%s",
		c.creds.AccessKey, scope, strings.Join(signedHeaders, ";"), signature,
	)

	// Build from the base URL (host intact), then set Opaque so
	// RequestURI sends the raw path verbatim, bypassing EscapedPath's
	// re-encoding of the colon.
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL, strings.NewReader(string(body)))
	if err != nil {
		return nil, err
	}
	req.URL.Opaque = rawPath
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	req.Header.Set("Authorization", authorization)

	resp, err := c.http.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("bedrock: %s: %s", resp.Status, truncate(string(raw), 300))
	}
	return raw, nil
}

func sha256Hex(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

func hmacSHA256(key []byte, data string) []byte {
	h := hmac.New(sha256.New, key)
	_, _ = h.Write([]byte(data))
	return h.Sum(nil)
}

func sortedKeys(m map[string]string) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}
