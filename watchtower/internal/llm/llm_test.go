package llm

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestChatJSONRoundTrip(t *testing.T) {
	var gotBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer key" {
			t.Errorf("auth = %q", r.Header.Get("Authorization"))
		}
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Error(err)
		}
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"{\"verdict\":\"fail\"}"}}]}`))
	}))
	defer srv.Close()

	c := New("gemini-2.5-flash", "key", srv.URL)
	var out struct {
		Verdict string `json:"verdict"`
	}
	if err := c.ChatJSON(context.Background(), "sys", "user", &out); err != nil {
		t.Fatalf("ChatJSON: %v", err)
	}
	if out.Verdict != "fail" {
		t.Fatalf("verdict = %q", out.Verdict)
	}
	if gotBody["model"] != "gemini-2.5-flash" {
		t.Errorf("model = %v", gotBody["model"])
	}
	msgs := gotBody["messages"].([]any)
	if len(msgs) != 2 || msgs[0].(map[string]any)["role"] != "system" {
		t.Errorf("messages = %v", msgs)
	}
	if gotBody["temperature"].(float64) != 0 {
		t.Errorf("temperature = %v", gotBody["temperature"])
	}
}

func TestChatJSONErrors(t *testing.T) {
	tests := []struct {
		name   string
		status int
		body   string
		want   string
	}{
		{name: "server error", status: 500, body: `{"error":"boom"}`, want: "500"},
		{name: "empty choices", status: 200, body: `{"choices":[]}`, want: "no choices"},
		{name: "model json garbage", status: 200, body: `{"choices":[{"message":{"content":"not json"}}]}`, want: "invalid JSON"},
		{name: "response not json", status: 200, body: `oops`, want: "decode response"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(tt.status)
				_, _ = w.Write([]byte(tt.body))
			}))
			defer srv.Close()
			c := New("m", "k", srv.URL)
			var out map[string]any
			err := c.ChatJSON(context.Background(), "s", "u", &out)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("err = %v, want containing %q", err, tt.want)
			}
		})
	}
}
