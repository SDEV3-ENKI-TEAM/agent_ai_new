# process_trace_json_openai_summary.py
import os
import json
import re
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

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

# 환경변수
PROCESSED_IDS_PATH = os.getenv("PROCESSED_IDS_PATH", "processed_ids.json")
TRACE_PATH = os.getenv("TRACE_JSON_PATH", "trace.json")  # 로컬
PERSIST_DIR = os.getenv("CHROMA_DIR", "./chroma_db")

EMB_COLL_NAME = os.getenv("CHROMA_EMB_COLLECTION", "trace_embeddings")
MAN_COLL_NAME = os.getenv("CHROMA_MAN_COLLECTION", "trace_manifests")

CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")


def _basename(path: Any):  # str
    return os.path.basename(path).lower() if isinstance(path, str) else ""


def _exe_from_cmd(cmd: Any):  # str
    if not isinstance(cmd, str):
        return ""
    m = re.search(r"([A-Za-z0-9_.-]+\.exe)", cmd, re.IGNORECASE)
    return m.group(1).lower() if m else _basename(cmd)


def _to_int(v: Any):
    try:
        return int(v)
    except Exception:
        return None


def extract_fields(span_src: Dict[str, Any]):  # Dict[str, Any]
    """f
    Jaeger → OpenSearch에 저장된 스팬 문서(_source)에서 필요한 필드만 추출
    """
    tag = span_src.get("tag", {}) if isinstance(span_src, dict) else {}
    start_ms = span_src.get("startTimeMillis") or span_src.get("startTime")
    start_ms = _to_int(start_ms)

    return {
        "traceID": span_src.get("traceID"),
        "spanID": span_src.get("spanID"),
        "operationName": span_src.get("operationName"),
        "startTimeMillis": start_ms,
        "duration": _to_int(span_src.get("duration")),
        "EventName": tag.get("EventName"),
        "EventID": tag.get("ID"),
        "Image": tag.get("Image"),
        "ImageBase": _basename(tag.get("Image")),
        "CommandLine": tag.get("CommandLine"),
        "CommandExe": _exe_from_cmd(tag.get("CommandLine")),
        "User": tag.get("User"),
        "SigmaAlert": tag.get("sigma@alert") or tag.get("SigmaAlert"),
        "SigmaStatus": tag.get("otel@status_description") or tag.get("SigmaStatus"),
        "raw": span_src,
    }


def span_references(span_src: Dict[str, Any]):
    refs = span_src.get("references", [])
    return refs if isinstance(refs, list) else []


# chunk 단위로 parent-child 관계를 묶고 시간순 정렬
def build_chunks_for_trace(
    spans: List[Dict[str, Any]],
):  # -> List[Tuple[str, List[Dict[str, Any]]]]
    """
    parentSpanID(anchor) → [parent?, children...] 묶고 시간순 정렬
    반환: [(anchor_span_id, [sorted spans...]), ...]
    """
    by_id = {s.get("spanID"): s for s in spans}
    children_by_parent: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    # 자식 그룹핑
    for s in spans:
        for r in span_references(s):
            if r.get("refType") == "CHILD_OF":
                p = r.get("spanID")
                if p:
                    children_by_parent[p].append(s)

    chunks: List[Tuple[str, List[Dict[str, Any]]]] = []
    for parent_id, childs in children_by_parent.items():
        seq: List[Dict[str, Any]] = []
        parent = by_id.get(parent_id)
        if parent:
            seq.append(parent)
        seq.extend(childs)
        seq.sort(key=lambda x: x.get("startTimeMillis") or 0)
        chunks.append((parent_id, seq))

    return chunks


# 요약 입력
def chunk_to_summary_source_lines(spans: List[Dict[str, Any]]):  # List[str]
    """
    (LLM에 줄) 간단한 한 줄 포맷으로 변환
    예: t=1753790415770 evt=ProcessCreate img=cmd.exe cmd=cmd.exe user=RUFFY\\임현빈 sigma=WSL Child Process Anomaly
    """
    lines: List[str] = []
    for s in spans:
        f = extract_fields(s)
        line = (
            f"t={f['startTimeMillis']} "
            f"evt={f['EventName'] or f['operationName'] or f['EventID']} "
            f"img={f['ImageBase']} cmd={f['CommandExe']} user={f['User']} "
            f"sigma={f['SigmaAlert'] or '-'}"
        )
        lines.append(line)
    return lines


# llm 요약
def summarize_chunk_korean(lines: List[str]):  # str

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY가 설정되어 있지 않습니다. .env를 확인하세요."
        )
    client = OpenAI()  # 키는 환경변수에서 자동 로드

    content = (
        "다음은 보안 이벤트 시퀀스입니다(시간 오름차순). "
        "핵심 행위를 2~5줄로 간단명료하게 한국어 요약을 써주세요. "
        "의심 신호(예: Sigma Alert)가 있으면 짧게 표시하고, 실행 파일명/부모-자식 흐름을 드러내 주세요.\n\n"
        "=== 이벤트 시퀀스 ===\n" + "\n".join(lines)
    )
    
    # return ( f"사용자 '{parts.get('user', '알 수 없음')}'가 '{parts.get('cmd', '알 수 없음')}' 명령어를 실행했습니다. " f"이벤트 ID는 {parts.get('evt', '없음')}이며, 공격 유형은 '{parts.get('attack', '없음')}'입니다." )

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": "당신은 침해대응 분석가입니다. 최대한 자세하고 세부적이면서도 간결하게 요약해서 알려주세요",
            },
            {"role": "user", "content": content},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


# 트레이스 아이디 중복 처리
def load_processed_ids(path: str = PROCESSED_IDS_PATH):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return set(json.load(f))
        except Exception:
            return set()
    return set()


def save_processed_ids(ids: set, path: str = PROCESSED_IDS_PATH):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sorted(ids), f, ensure_ascii=False, indent=2)


# 메인 파이프라인
def process_trace_file(
    json_path: str = TRACE_PATH,
    persist_dir: str = PERSIST_DIR,
    emb_coll_name: str = EMB_COLL_NAME,
    man_coll_name: str = MAN_COLL_NAME,
):
    """
    1 trace.json 로드/정제
    2 traceID 단위 그룹핑 / opensearch에서 가져올때 traceID 단위 기반 그룹핑 시도
    3 각 trace에서 parent-child 기반 청크 생성
    4 청크를 LLM으로 요약 → 임베딩 → trace_embeddings에 저장
    5 traceID → [chunk_id ~] 매니페스트를 trace_manifests에 저장
    6 processed_ids.json에 traceID 기록 -> 중복 제거
    """
    raw = load_any_json(json_path)
    hits = coerce_to_hits(raw)

    traces: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for h in hits:
        src = h.get("_source", {})
        tid = src.get("traceID")
        if tid:
            traces[tid].append(src)

    if not traces:
        print("traceID가 포함된 데이터가 없습니다.")
        return

    processed_ids = load_processed_ids()
    client = init_chroma(persist_directory=persist_dir)

    emb_coll = get_or_create_collection(
        client, emb_coll_name, metadata={"kind": "trace-chunk-embeddings"}
    )
    man_coll = get_or_create_collection(
        client, man_coll_name, metadata={"kind": "trace-manifest"}
    )

    total_chunks = 0
    newly_processed = set()

    for trace_id, span_srcs in traces.items():
        if trace_id in processed_ids:
            print(f"이미 처리됨: traceID={trace_id}")
            continue

        # startTimeMillis -> 없을 경우 startTime
        spans: List[Dict[str, Any]] = []
        for s in span_srcs:
            if "startTimeMillis" not in s and isinstance(s.get("startTime"), int):
                s["startTimeMillis"] = s["startTime"]
            spans.append(s)

        chunks = build_chunks_for_trace(spans)
        if not chunks:
            # 자식 관계가 전혀 없으면 전체를 1개 청크로
            spans_sorted = sorted(spans, key=lambda x: x.get("startTimeMillis") or 0)
            root_anchor = spans_sorted[0].get("spanID") or "root"
            chunks = [(root_anchor, spans_sorted)]

        chunk_ids: List[str] = []

        for i, (anchor_id, seq) in enumerate(chunks):
            # 요약 생성 -> anchor_id는 청크 대표 id(부모 id)
            lines = chunk_to_summary_source_lines(seq)
            summary = summarize_chunk_korean(lines)

            # 임베딩 생성
            embeddings = openai_embed_texts([summary])

            # 메타데이터 생성
            has_alert = any(
                (
                    extract_fields(s).get("SigmaAlert") is not None
                    and str(extract_fields(s).get("SigmaAlert")).strip()
                )
                for s in seq
            )
            meta = {
                "traceID": trace_id,
                "anchorSpanID": anchor_id,
                "chunkIndex": i,
                "spanCount": len(seq),
                "hasAlert": bool(has_alert),
            }

            # Chroma 저장
            doc_id = f"{trace_id}:{anchor_id}:{i:03d}"
            add_embeddings(
                emb_coll,
                ids=[doc_id],
                embeddings=embeddings,
                metadatas=[meta],
                documents=[summary],
            )
            chunk_ids.append(doc_id)
            total_chunks += 1

        # 매니페스트 저장 (traceID → chunk_ids)
        manifest_doc = json.dumps(
            {"traceID": trace_id, "chunks": chunk_ids}, ensure_ascii=False
        )
        try:
            man_coll.delete(ids=[trace_id])  # 이미 있으면 덮어쓰기
        except Exception:
            pass
        man_coll.add(
            ids=[trace_id],
            documents=[manifest_doc],
            metadatas=[{"traceID": trace_id, "chunkCount": len(chunk_ids)}],
        )

        newly_processed.add(trace_id)
        print(f"[OK] traceID={trace_id} → {len(chunk_ids)}개 청크 저장")

    if newly_processed:
        processed_ids |= newly_processed
        save_processed_ids(processed_ids)

    print(
        f"완료: 신규 청크 {total_chunks}개 저장, 총 처리 traceID {len(newly_processed)}개"
    )


# 저장된 데이터 점검
def inspect_chroma(
    persist_dir: str = PERSIST_DIR, 
    emb_coll_name: str = EMB_COLL_NAME,
    man_coll_name: str = MAN_COLL_NAME,
    sample_trace_id: Optional[str] = None,
    show_vectors: bool = False,
):
    # 저장된 매니페스트/청크 요약/메타/벡터를 출력 ->  빠르게 확인
    client = init_chroma(persist_directory=persist_dir)
    emb = get_or_create_collection(client, emb_coll_name)
    man = get_or_create_collection(client, man_coll_name)

    print("\n[COLLECTIONS]")
    print(" -", emb.name, "(embeddings)")
    print(" -", man.name, "(manifests)")

    man_all = man.get()
    print(f"[MANIFEST COUNT] {len(man_all.get('ids', []))}")

    manifest_map = {}
    for mid, mdoc in zip(man_all.get("ids", []), man_all.get("documents", [])):
        try:
            manifest_map[mid] = json.loads(mdoc)
        except Exception:
            pass

    if not manifest_map:
        print("매니페스트가 비어 있습니다.")
        return

    if not sample_trace_id:
        sample_trace_id = next(iter(manifest_map.keys()))
    print(f"\n[MANIFEST for traceID={sample_trace_id}]")
    print(json.dumps(manifest_map[sample_trace_id], ensure_ascii=False, indent=2))

    chunk_ids = manifest_map[sample_trace_id]["chunks"]
    include = ["documents", "metadatas"] + (["embeddings"] if show_vectors else [])
    got = emb.get(ids=chunk_ids, include=include)

    for i, cid in enumerate(got.get("ids", [])):
        print(f"\n- CHUNK {i} / ID={cid}")
        print("  META:", got["metadatas"][i])
        print(
            "  DOC :\n",
            got["documents"][i][:500],
            ("..." if len(got["documents"][i]) > 500 else ""),
        )
        if show_vectors and got.get("embeddings"):
            vec = got["embeddings"][i]
            print("  EMBEDDING: dim=", len(vec), " (앞 5개)", vec[:5])


if __name__ == "__main__":
    # 전처리 -> 요약(LLM) -> 임베딩 -> Chroma 저장
    process_trace_file()

    # 테스트
    inspect_chroma(show_vectors=False)
