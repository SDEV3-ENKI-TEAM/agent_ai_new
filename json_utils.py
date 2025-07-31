# json_utils.py
import json
import re
from typing import Any, List, Dict


def _try_json(text: str):
    try:
        return json.loads(text)
    except Exception:
        return None


def _strip_bom(s: str) -> str:
    # UTF-8 BOM 제거
    if s and s.startswith("\ufeff"):
        return s.lstrip("\ufeff")
    return s


def _fix_trailing_commas(text: str):  # str
    # } 또는 ] 앞의 마지막 , 제거 (간단 휴리스틱)
    return re.sub(r",\s*([\]}])", r"\1", text)


def _replace_single_quotes(text: str):  # str
    # 전체 ' -> " 교체
    return re.sub(r"'", r'"', text)


def load_any_json(path: str):  # any
    """
    path의 내용을 읽어 가능한 한 JSON으로 변환
    1) 정상 JSON
    2) trailing comma 제거
    3) ' -> " 전환
    4) NDJSON 라인별 파싱
    실패 시 예외
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    raw = _strip_bom(raw)

    data = _try_json(raw)
    if data is not None:
        return data

    data = _try_json(_fix_trailing_commas(raw))
    if data is not None:
        return data

    data = _try_json(_replace_single_quotes(raw))
    if data is not None:
        return data

    # NDJSON
    objs = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        obj = _try_json(line)
        if obj is not None:
            objs.append(obj)
        else:
            obj = _try_json(_fix_trailing_commas(line))
            if obj is not None:
                objs.append(obj)
    if objs:
        return objs

    raise ValueError(f"파일을 JSON으로 해석할 수 없습니다: {path}")


def coerce_to_hits(data: Any):  # List[Dict]
    """
    다양한 형태를 'OpenSearch hit 유사 리스트'로 통일.
    - 리스트: 각 원소가 dict라고 가정. _source 없으면 "{ _source: item }" 로 래핑
    - dict:
        * data 키(리스트) 있으면 그걸 사용
        * hits.hits 있으면 그 리스트 사용
        * 아니면 dict 자체를 단일 원소로 래핑
    반환: 각 원소는 dict이며 내부에 _source가 존재하도록 통일
    """
    hits: List[Dict] = []

    def wrap_hit(obj: Dict):  # Dict
        return obj if "_source" in obj else {"_source": obj}

    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                hits.append(item if "_source" in item else {"_source": item})
    elif isinstance(data, dict):
        if "data" in data and isinstance(data["data"], list):
            for item in data["data"]:
                if isinstance(item, dict):
                    hits.append(wrap_hit(item))
        elif (
            "hits" in data and isinstance(data["hits"], dict) and "hits" in data["hits"]
        ):
            for item in data["hits"]["hits"]:
                if isinstance(item, dict):
                    hits.append(item if "_source" in item else {"_source": item})
        else:
            hits.append(wrap_hit(data))
    else:
        raise ValueError("지원되지 않는 JSON 최상위 타입입니다.")

    return hits
