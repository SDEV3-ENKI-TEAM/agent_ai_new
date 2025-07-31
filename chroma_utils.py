# chroma_utils.py
import os
import chromadb
from chromadb.config import Settings, DEFAULT_TENANT, DEFAULT_DATABASE
from chromadb.errors import NotFoundError
from typing import Any, Dict, List, Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def init_chroma(persist_directory: str = "./chroma_db"):  # chromadb.PersistentClient
    # chroma PersistentClient 초기화
    os.makedirs(persist_directory, exist_ok=True)
    client = chromadb.PersistentClient(
        path=persist_directory,
        settings=Settings(),
        tenant=DEFAULT_TENANT,
        database=DEFAULT_DATABASE,
    )
    return client


def get_or_create_collection(
    client: chromadb.PersistentClient,
    name: str,
    metadata: Optional[Dict[str, Any]] = None,
):
    # 컬렉션을 가져오거나 없으면 생성
    # (임베딩은 우리가 직접 생성해서 add 시 embeddings 파라미터로 전달)
    try:
        return client.get_collection(name)
    except NotFoundError:
        return client.create_collection(name=name, metadata=metadata)


def openai_embed_texts(texts: List[str]):  # List[List[float]]
    # OpenAI Embeddings API로 문서 배열 -> 벡터 배열
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY가 설정되어 있지 않습니다 (.env 확인).")
    model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    client = OpenAI()  # 환경변수에서 키 자동 로드
    resp = client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in resp.data]


def add_embeddings(
    collection,
    ids: List[str],
    embeddings: List[List[float]],
    metadatas: Optional[List[Dict[str, Any]]] = None,
    documents: Optional[List[str]] = None,
):
    """
    미리 생성한 임베딩(벡터)을 컬렉션에 저장
    """
    collection.add(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=documents,
    )
