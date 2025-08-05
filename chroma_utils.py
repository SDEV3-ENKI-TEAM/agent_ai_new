# chroma_utils.py
import os
import openai
import chromadb
from chromadb.config import Settings, DEFAULT_TENANT, DEFAULT_DATABASE
from chromadb.utils import embedding_functions
from chromadb.errors import NotFoundError


def init_chroma(persist_directory: str = "./chroma_db"):
    os.makedirs(persist_directory, exist_ok=True)
    client = chromadb.PersistentClient(
        path=persist_directory,
        settings=Settings(),
        tenant=DEFAULT_TENANT,
        database=DEFAULT_DATABASE,
    )
    return client


def get_or_create_collection(
    client: chromadb.PersistentClient, name: str, metadata: dict = None
):
    try:
        return client.get_collection(name)
    except NotFoundError:
        return client.create_collection(
            name=name,
            metadata=metadata or {},
            embedding_function=embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.getenv("OPENAI_API_KEY"), model_name="text-embedding-ada-002"
            ),
        )


def openai_embed_texts(texts: list[str], model: str = "text-embedding-ada-002"):
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY가 설정되어 있지 않습니다.")
    resp = openai.embeddings.create(model=model, input=texts)
    return [c.embedding for c in resp.data]


def add_embeddings(
    collection,
    ids: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict],
    documents: list[str],
):
    collection.add(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=documents,
    )
