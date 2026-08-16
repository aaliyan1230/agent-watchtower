package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"time"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracehttp"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/trace"
)

// runTrace exports one agent run over OTLP/HTTP and fetches its
// verdict. One export request = one run: the batch processor is
// flushed explicitly after the span tree ends, mirroring the harness.
func runTrace(endpoint string, variant string) (map[string]any, error) {
	ctx := context.Background()
	spec, _ := buildSpec(variant)

	hostPort, err := hostPort(endpoint)
	if err != nil {
		return nil, err
	}
	exporter, err := otlptracehttp.New(ctx,
		otlptracehttp.WithEndpoint(hostPort),
		otlptracehttp.WithInsecure(),
		otlptracehttp.WithURLPath("/v1/traces"),
	)
	if err != nil {
		return nil, fmt.Errorf("exporter: %w", err)
	}
	provider := sdktrace.NewTracerProvider(
		sdktrace.WithBatcher(exporter),
	)
	tracer := provider.Tracer("watchtower.goproducer")

	ctx, rootSpan := tracer.Start(ctx, spec.Name)
	traceID := rootSpan.SpanContext().TraceID().String()
	applyAttrs(rootSpan, spec.Attrs)
	for _, event := range spec.Events {
		rootSpan.AddEvent(event.Name, trace.WithAttributes(toKeyValues(event.Attr)...))
	}
	for _, child := range spec.Children {
		emit(ctx, tracer, child)
	}
	rootSpan.End()

	if err := provider.ForceFlush(ctx); err != nil {
		return nil, fmt.Errorf("flush: %w", err)
	}
	if err := provider.Shutdown(ctx); err != nil {
		return nil, fmt.Errorf("shutdown: %w", err)
	}
	return fetchReport(endpoint, traceID)
}

func emit(ctx context.Context, tracer trace.Tracer, spec *spanSpec) {
	ctx, span := tracer.Start(ctx, spec.Name)
	applyAttrs(span, spec.Attrs)
	for _, event := range spec.Events {
		span.AddEvent(event.Name, trace.WithAttributes(toKeyValues(event.Attr)...))
	}
	for _, child := range spec.Children {
		emit(ctx, tracer, child)
	}
	span.End()
}

func applyAttrs(span trace.Span, attrs map[string]any) {
	span.SetAttributes(toKeyValues(attrs)...)
}

func toKeyValues(attrs map[string]any) []attribute.KeyValue {
	out := make([]attribute.KeyValue, 0, len(attrs))
	for key, value := range attrs {
		out = append(out, attribute.KeyValue{Key: attribute.Key(key), Value: toAttrValue(value)})
	}
	return out
}

func toAttrValue(value any) attribute.Value {
	switch v := value.(type) {
	case string:
		return attribute.StringValue(v)
	case bool:
		return attribute.BoolValue(v)
	case int:
		return attribute.IntValue(v)
	case int64:
		return attribute.Int64Value(v)
	case float64:
		return attribute.Float64Value(v)
	default:
		return attribute.StringValue(fmt.Sprintf("%v", v))
	}
}

func hostPort(endpoint string) (string, error) {
	u, err := url.Parse(endpoint)
	if err != nil {
		return "", fmt.Errorf("parse endpoint: %w", err)
	}
	if u.Host == "" {
		return "", fmt.Errorf("endpoint %q has no host", endpoint)
	}
	return u.Host, nil
}

func fetchReport(endpoint, traceID string) (map[string]any, error) {
	url := endpoint + "/v1/reports/" + traceID
	var lastStatus int
	for attempt := 0; attempt < 5; attempt++ {
		resp, err := http.Get(url)
		if err == nil {
			body, readErr := io.ReadAll(resp.Body)
			resp.Body.Close()
			lastStatus = resp.StatusCode
			if resp.StatusCode == http.StatusOK {
				if readErr != nil {
					return nil, fmt.Errorf("read report: %w", readErr)
				}
				var report map[string]any
				if jsonErr := json.Unmarshal(body, &report); jsonErr != nil {
					return nil, fmt.Errorf("decode report: %w", jsonErr)
				}
				return report, nil
			}
		}
		time.Sleep(100 * time.Millisecond)
	}
	return nil, fmt.Errorf("no report for trace %s after retries (last status %d)", traceID, lastStatus)
}
