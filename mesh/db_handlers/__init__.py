from .base import VectorDB
from .milvus import MilvusDB
from .qdrant import QdrantDB

db_instances = {"milvus": MilvusDB, "qdrant": QdrantDB}

def db_factory(db_type: str) -> VectorDB:
    return db_instances[db_type]