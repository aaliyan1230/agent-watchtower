.PHONY: test vet serve demo live venv producer cross-producer experiment experiment-evidence experiment-false-assurance experiment-fa-live experiment-live experiment-bedrock experiment-agreement diagrams

test: ## go test -race (core + producer) + pytest
	cd watchtower && go test ./... -race
	cd producers/goproducer && go test ./...
	.venv/bin/pytest tests -q

diagrams: ## regenerate diagram PNGs from .excalidraw sources
	./scripts/diagrams/export.sh

vet: ## go vet ./... (core + producer)
	cd watchtower && go vet ./...
	cd producers/goproducer && go vet ./...

serve: ## run watchtower serve on :4318
	cd watchtower && go run ./cmd/watchtower serve --addr :4318

demo: ## offline python demo: fake provider -> spans -> serve -> report
	.venv/bin/python -m harness.demo

live: ## live smoke test (needs GEMINI_API_KEY in .env)
	.venv/bin/python -m harness.live_smoke

venv: ## create .venv and install harness + experiments
	python3 -m venv .venv
	.venv/bin/pip install -e .

producer: ## build the Go reference producer (independent OTel producer)
	cd producers/goproducer && go build -o /tmp/watchtower-goproducer .

cross-producer: ## Phase 2.9 acceptance: two producers, one evidence contract
	.venv/bin/python -m pytest tests/test_cross_producer.py -q

experiment: ## offline matrix run + analysis (deterministic, free)
	.venv/bin/python -m experiments.suite --out artifacts
	.venv/bin/python -m experiments.run --grid artifacts/grid.json --out artifacts/results.json
	.venv/bin/python -m experiments.analyze --results artifacts/results.json

experiment-evidence: ## clean-behavior telemetry-fault grid (deterministic, free)
	.venv/bin/python -m experiments.suite --evidence --out artifacts
	.venv/bin/python -m experiments.run --evidence --out artifacts/results-evidence.json
	.venv/bin/python -m experiments.analyze --results artifacts/results-evidence.json

experiment-false-assurance: ## 2x2 behavior x telemetry grid, contract vs behavior-only baseline
	.venv/bin/python -m experiments.run --false-assurance --no-traces --out artifacts/results-fa.json
	.venv/bin/python -m experiments.run --false-assurance --no-traces --config experiment_behavior_only_config.json --out artifacts/results-fa-baseline.json
	.venv/bin/python -m experiments.analyze --false-assurance --results artifacts/results-fa.json --baseline artifacts/results-fa-baseline.json

experiment-fa-live: ## live judge pilot over the four conditions (needs GEMINI_API_KEY in .env)
	.venv/bin/python -m experiments.run --live --false-assurance --pilot --out artifacts/results-fa-live.json
	.venv/bin/python -m experiments.analyze --false-assurance --results artifacts/results-fa-live.json

experiment-live: ## small live stratified sample (needs GEMINI_API_KEY in .env)
	.venv/bin/python -m experiments.run --live --pilot --out artifacts/results-live.json
	.venv/bin/python -m experiments.analyze --results artifacts/results-live.json

experiment-bedrock: ## same pilot with the Bedrock (Nova) judge (needs AWS CLI creds)
	.venv/bin/python -m experiments.run --live --pilot --judge bedrock --out artifacts/results-live-bedrock.json
	.venv/bin/python -m experiments.analyze --results artifacts/results-live-bedrock.json

experiment-agreement: ## inter-judge kappa across providers
	.venv/bin/python -m experiments.analyze --results artifacts/results-live.json --compare artifacts/results-live-bedrock.json --compare artifacts/results-live-kimi.json
