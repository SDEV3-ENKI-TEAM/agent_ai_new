# opensearch_utils.py
import os
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any
from opensearchpy import OpenSearch

KST = timezone(timedelta(hours=9))
FMT = "%Y-%m-%d %H:%M:%S"


def millis_to_str(ms: int):
    return datetime.fromtimestamp(ms / 1000, KST).strftime(FMT)


def os_client() -> OpenSearch:
    host = os.getenv("OS_HOST")
    port = int(os.getenv("OS_PORT", "443"))
    user = os.getenv("OS_USER")
    pwd = os.getenv("OS_PASS")
    if not all([host, user, pwd]):
        raise RuntimeError("OS_HOST/OS_PORT/OS_USER/OS_PASS 환경변수를 설정하세요.")
    return OpenSearch(
        hosts=[{"host": host, "port": port}],
        http_auth=(user, pwd),
        use_ssl=True,
        verify_certs=True,
    )


INDEX = os.getenv("OS_INDEX", "jaeger-span-*")


def list_recent_trace_ids(
    client: OpenSearch, lookback: str = "now-10s", limit: int = 200
):
    """최근 lookback~now 구간에서 최신 TraceID N개"""
    body = {
        "size": 0,
        "query": {"range": {"startTimeMillis": {"gte": lookback, "lte": "now"}}},
        "aggs": {
            "traces": {
                "terms": {
                    "field": "traceID",
                    "size": limit,
                    "order": {"max_start": "desc"},
                },
                "aggs": {"max_start": {"max": {"field": "startTimeMillis"}}},
            }
        },
    }
    res = client.search(index=INDEX, body=body)
    return [b["key"] for b in res["aggregations"]["traces"]["buckets"]]


def get_trace_spans(client: OpenSearch, trace_id: str):
    """특정 TraceID의 스팬 전체(시간순)"""
    body = {
        "size": 1000,
        "query": {"term": {"traceID": trace_id}},
        "sort": [{"startTimeMillis": "asc"}],
    }
    res = client.search(index=INDEX, body=body)
    spans = []
    for hit in res["hits"]["hits"]:
        s = hit["_source"]
        # downstream 코드가 startTime 문자열을 기대하므로 추가
        if "startTimeMillis" in s and "startTime" not in s:
            s["startTime"] = millis_to_str(s["startTimeMillis"])
        spans.append(s)
    return {"trace_id": trace_id, "span_count": len(spans), "spans": spans}
