package model

// Watchtower-specific attribute added here (rather than in commit 1's
// semconv.go) because it earns its keep once the loop verifier needs it.
const WatchtowerToolArgs = "watchtower.tool.args" // ours: JSON string of tool arguments
