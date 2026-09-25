from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

_LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60)


class Metrics:
    """Prometheus metrics, on a registry per app so tests can build many apps."""

    def __init__(self) -> None:
        r = self.registry = CollectorRegistry()
        self.http_requests = Counter(
            "sceneid_http_requests_total",
            "HTTP requests",
            ["method", "route", "status"],
            registry=r,
        )
        self.http_latency = Histogram(
            "sceneid_http_request_duration_seconds",
            "HTTP request latency",
            ["method", "route"],
            buckets=_LATENCY_BUCKETS,
            registry=r,
        )
        self.matches = Counter(
            "sceneid_matches_total",
            "Match requests by outcome: match, unknown, bad_input, busy, error",
            ["outcome"],
            registry=r,
        )
        self.stage_seconds = Histogram(
            "sceneid_match_stage_seconds",
            "Time spent in each matching stage",
            ["stage"],
            buckets=_LATENCY_BUCKETS,
            registry=r,
        )
        self.confidence = Histogram(
            "sceneid_match_confidence",
            "Confidence of each decision, accepted or not",
            buckets=[i / 20 for i in range(1, 21)],
            registry=r,
        )
        self.upload_bytes = Histogram(
            "sceneid_upload_bytes",
            "Size of uploaded clips",
            buckets=(1e5, 1e6, 5e6, 1e7, 2.5e7, 5e7, 1e8),
            registry=r,
        )
        self.inflight = Gauge("sceneid_matches_inflight", "Matches running now", registry=r)
        self.library_videos = Gauge("sceneid_library_videos", "Indexed videos", registry=r)
        self.library_vectors = Gauge("sceneid_library_vectors", "Indexed frame vectors", registry=r)
