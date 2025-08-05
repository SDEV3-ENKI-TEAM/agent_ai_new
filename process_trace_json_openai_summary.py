# process_trace_single_chunk.py
import os
import json
from datetime import datetime
from typing import Dict, Any, List, Set
import time
import pandas as pd
import argparse
from opensearch_utils import os_client, list_recent_trace_ids, get_trace_spans


from dotenv import load_dotenv
from openai import OpenAI

from json_utils import load_any_json, coerce_to_hits
from chroma_utils import (
    init_chroma,
    get_or_create_collection,
    openai_embed_texts,
    add_embeddings,
)

load_dotenv()


OS_LOOKBACK_SECONDS = int(os.getenv("OS_LOOKBACK_SECONDS", "30"))  # 초기 조회 범위
POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL_SECONDS", "5"))
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
EMB_COLL_NAME = os.getenv("CHROMA_EMB_COLLECTION", "trace_embeddings")
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
PROCESSED_FILE = os.getenv("PROCESSED_IDS_PATH", "processed_ids.json")
OS_INDEX = os.getenv("OS_INDEX", "jaeger-span-*")


def now_millis():
    return int(time.time() * 1000)


def load_processed():
    if os.path.exists(PROCESSED_FILE):
        try:
            return set(json.load(open(PROCESSED_FILE, "r", encoding="utf-8")))
        except:
            return set()
    return set()


def save_processed(ids: Set[str]):
    with open(PROCESSED_FILE, "w", encoding="utf-8") as f:
        json.dump(sorted(ids), f, ensure_ascii=False, indent=2)


# def save_processed_ids(ids: set, path=PROCESSED_FILE):
#     with open(path, "w", encoding="utf-8") as f:
#         json.dump(sorted(ids), f, ensure_ascii=False, indent=2)


def list_all_trace_ids(client, limit=1000):
    body = {
        "size": 0,
        "aggs": {"traces": {"terms": {"field": "traceID", "size": limit}}},
    }
    res = client.search(index=OS_INDEX, body=body)
    return [
        b["key"]
        for b in res.get("aggregations", {}).get("traces", {}).get("buckets", [])
    ]


def summarize_with_llm(text: str):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY 환경변수를 설정하세요.")
    client = OpenAI()
    system_prompt = "당신은 침해사고 대응 전문가입니다. 아래 보안 이벤트 시퀀스를 2~3문장으로 한 단락 요약하되, "
    user_prompt = f"{text}"
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


def process_trace(trace_id: str, spans: List[Dict[str, Any]], emb_coll):
    # DataFrame 전처리
    rows = []
    for s in spans:
        tag = s.get("tag", {})
        rows.append(
            {
                "User": tag.get("User"),
                "CommandLine": tag.get("CommandLine"),
                "EventName": tag.get("EventName"),
                "SigmaAlert": tag.get("sigma@alert") or tag.get("SigmaAlert"),
                "startTime": s.get("startTime"),
            }
        )
    df = pd.DataFrame(rows).sort_values("startTime")

    # 요약 텍스트 구성
    lines = [
        f"사용자 '{r['User']}'이 '{r['CommandLine']}'을 실행, 이벤트 '{r['EventName']}', 경고: {r['SigmaAlert']}"
        for _, r in df.iterrows()
    ]
    raw_text = "\n".join(lines)
    summary = summarize_with_llm(raw_text)

    print(f"TraceID={trace_id} 요약 결과:\n{summary}\n")

    # 임베딩 생성
    vector = openai_embed_texts([summary])[0]

    # Chroma에 저장: id=traceID, metadata에 traceID, summary, spanCount 포함
    add_embeddings(
        emb_coll,
        ids=[trace_id],
        embeddings=[vector],
        metadatas=[{"traceID": trace_id, "spanCount": len(spans), "summary": summary}],
        documents=[summary],
    )
    print(f"Processed trace {trace_id}, spans={len(spans)}")


# 초기 처리 및 스트리밍 폴링
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--force", action="store_true", help="모든 trace 강제 재처리"
    )
    args = parser.parse_args()

    processed = set() if args.force else load_processed()
    client_os = os_client()
    client_chroma = init_chroma(CHROMA_DIR)
    emb_coll = get_or_create_collection(
        client_chroma, EMB_COLL_NAME, metadata={"kind": "streaming"}
    )

    # 기존 트레이스 처리
    print("Initializing: processing all existing traces...")
    all_ids = list_all_trace_ids(client_os, limit=1000)
    for tid in all_ids:
        if not args.force and tid in processed:
            continue
        spans = get_trace_spans(client_os, tid)["spans"]
        process_trace(tid, spans, emb_coll)
        processed.add(tid)
    save_processed(processed)

    last_ts = now_millis()
    print("Entering streaming poll loop...")

    try:
        while True:
            current_ts = now_millis()
            body = {
                "size": 0,
                "query": {
                    "range": {"startTimeMillis": {"gte": last_ts, "lte": current_ts}}
                },
                "aggs": {"traces": {"terms": {"field": "traceID", "size": 1000}}},
            }
            res = client_os.search(index=OS_INDEX, body=body)
            changed_ids = [
                b["key"]
                for b in res.get("aggregations", {})
                .get("traces", {})
                .get("buckets", [])
            ]
            if changed_ids:
                print(f"Detected changed traces: {changed_ids}")
            for tid in changed_ids:
                if not args.force and tid in processed:
                    continue
                spans = get_trace_spans(client_os, tid)["spans"]
                process_trace(tid, spans, emb_coll)
                processed.add(tid)
            if changed_ids:
                save_processed(processed)
            last_ts = current_ts
            time.sleep(POLL_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        print("Processing stopped.")


if __name__ == "__main__":
    main()
