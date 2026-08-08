// Command watchtower is the verification CLI: it turns span traces into
// evidence-backed verdicts. Two modes:
//
//	watchtower verify --trace-file X --config Y   one-shot verdict for a trace file
//	watchtower serve --addr :4318                ingest endpoint that verifies live
package main

import "os"

func main() {
	if err := NewRootCmd().Execute(); err != nil {
		os.Exit(1)
	}
}
