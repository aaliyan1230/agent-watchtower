// Package main is the Go reference producer for the cross-producer
// evidence contract: it emits one agent run as OTel GenAI spans using
// the official Go OTel SDK, exports them over OTLP/HTTP, and fetches
// the Watchtower verdict for the trace.
//
// It shares no code with the Python harness: different language,
// different SDK, same wire contract. Its job in the research pipeline
// is to prove that the claim-specific evidence obligations are
// producer-neutral — a complete run gets PASS, and a run whose
// producer omits a required field gets INCONCLUSIVE, never a silently
// weakened PASS.
package main

import (
	"flag"
	"fmt"
	"log"
	"os"
)

const defaultEndpoint = "http://127.0.0.1:4318"

func main() {
	endpoint := flag.String("endpoint", defaultEndpoint, "watchtower serve base URL")
	variant := flag.String("variant", "clean", "trace variant: clean, omit-genai-system, omit-tool-result, omit-tool-id, omit-final")
	flag.Parse()

	if _, ok := buildSpec(*variant); !ok {
		log.Fatalf("unknown variant %q (want one of clean, omit-genai-system, omit-tool-result, omit-tool-id, omit-final)", *variant)
	}
	report, err := runTrace(*endpoint, *variant)
	if err != nil {
		log.Fatalf("goproducer: %v", err)
	}
	verdict, _ := report["verdict"].(string)
	fmt.Printf("verdict=%s protocol=%v traceId=%v\n", verdict, report["protocolVersion"], report["traceId"])
	if verdict == "" {
		os.Exit(2)
	}
}
