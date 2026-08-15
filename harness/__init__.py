"""harness: Python agent layer for Watchtower.

Emits OTel GenAI spans describing agent runs; the Go service ingests,
reconstructs, verifies, and returns an evidence-backed verdict.
"""

from .providers import FakeProvider, GeminiProvider, Provider, ProviderResponse
from .faults import EvidenceFaultKind, FaultInjector, FaultKind, FaultSpec
from .telemetry import EvidenceFaultExporter, HarnessTelemetry
from .workers import Tool, Worker
from .supervisor import Supervisor

__all__ = [
    "FakeProvider",
    "GeminiProvider",
    "Provider",
    "ProviderResponse",
    "FaultInjector",
    "FaultKind",
    "EvidenceFaultKind",
    "FaultSpec",
    "EvidenceFaultExporter",
    "HarnessTelemetry",
    "Tool",
    "Worker",
    "Supervisor",
]
