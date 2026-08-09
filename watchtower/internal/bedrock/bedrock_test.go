package bedrock

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func testClient(t *testing.T, handler http.HandlerFunc) (*Client, *httptest.Server) {
	t.Helper()
	srv := httptest.NewServer(handler)
	c := New("amazon.nova-lite-v1:0", "us-east-1", Credentials{
		AccessKey: "AKID", SecretKey: "SECRET", SessionToken: "TOKEN",
	})
	c.baseURL = srv.URL
	return c, srv
}

func TestChatJSONRequestShape(t *testing.T) {
	var gotBody map[string]any
	var authHeader, dateHeader, requestURI string
	c, srv := testClient(t, func(w http.ResponseWriter, r *http.Request) {
		authHeader = r.Header.Get("Authorization")
		dateHeader = r.Header.Get("X-Amz-Date")
		requestURI = r.RequestURI
		_ = json.NewDecoder(r.Body).Decode(&gotBody)
		_, _ = w.Write([]byte(`{"output":{"message":{"content":[{"text":"{\"verdict\":\"ok\"}"}]}}}`))
	})
	defer srv.Close()

	text, err := c.ChatText(context.Background(), "sys", "user")
	if err != nil {
		t.Fatalf("ChatText: %v", err)
	}
	var out struct {
		Verdict string `json:"verdict"`
	}
	if err := json.Unmarshal([]byte(text), &out); err != nil {
		t.Fatalf("parse: %v", err)
	}
	if out.Verdict != "ok" {
		t.Fatalf("verdict = %q", out.Verdict)
	}
	// The wire path carries the literal colon; the signed canonical
	// URI (inside the signature) percent-encodes it.
	if requestURI != "/model/amazon.nova-lite-v1:0/invoke" {
		t.Errorf("request URI = %q", requestURI)
	}
	if !strings.HasPrefix(authHeader, "AWS4-HMAC-SHA256 Credential=AKID/") {
		t.Errorf("authorization = %q", authHeader)
	}
	if !strings.Contains(authHeader, "/us-east-1/bedrock/aws4_request, SignedHeaders=") {
		t.Errorf("authorization scope = %q", authHeader)
	}
	if !strings.Contains(authHeader, "Signature=") {
		t.Errorf("authorization missing signature: %q", authHeader)
	}
	if dateHeader == "" {
		t.Error("missing x-amz-date")
	}
	if !strings.Contains(authHeader, "x-amz-security-token") {
		t.Errorf("session token not signed: %q", authHeader)
	}
	sys := gotBody["system"].([]any)[0].(map[string]any)
	if sys["text"] != "sys" {
		t.Errorf("system = %v", sys)
	}
	cfg := gotBody["inferenceConfig"].(map[string]any)
	if cfg["temperature"].(float64) != 0 {
		t.Errorf("temperature = %v", cfg["temperature"])
	}
}

func TestChatJSONErrors(t *testing.T) {
	tests := []struct {
		name   string
		status int
		body   string
		want   string
	}{
		{name: "server error", status: 500, body: `{"message":"boom"}`, want: "500"},
		{name: "empty response", status: 200, body: `{"output":{"message":{"content":[]}}}`, want: "empty response"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c, srv := testClient(t, func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(tt.status)
				_, _ = w.Write([]byte(tt.body))
			})
			defer srv.Close()
			_, err := c.ChatText(context.Background(), "s", "u")
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("err = %v, want containing %q", err, tt.want)
			}
		})
	}
}

func TestCredentialsFromEnv(t *testing.T) {
	t.Setenv("AWS_ACCESS_KEY_ID", "AK")
	t.Setenv("AWS_SECRET_ACCESS_KEY", "SK")
	t.Setenv("AWS_SESSION_TOKEN", "ST")
	c, err := CredentialsFromEnv()
	if err != nil || c.AccessKey != "AK" || c.SecretKey != "SK" || c.SessionToken != "ST" {
		t.Fatalf("creds = %+v, %v", c, err)
	}
	t.Setenv("AWS_ACCESS_KEY_ID", "")
	if _, err := CredentialsFromEnv(); err == nil {
		t.Fatal("missing key should fail loudly")
	}
}
