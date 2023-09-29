from .base import VectorDB
from .milvus import MilvusDB

db_instances = {"milvus": MilvusDB}

def db_factory(db_type: str) -> VectorDB:
    return db_instances[db_type]