import numpy as np

from .base import *
from typing import Optional, Dict
from functools import reduce
from pymilvus import connections, utility, Collection, FieldSchema, CollectionSchema, DataType

class MilvusDB(VectorDB):
    def __init__(self, host: str, port: int, alias: str, collection_str: str, collection: Optional[Collection] = None):
        self.host = host
        self.port = port
        self.alias = alias
        self.collection_str = collection_str

        if collection is not None:
            self.collection = collection
        else:
            self.collection = MilvusDB.__load_collection(host, port, alias, collection_str)
        self.collection.load()

    def connect(self):
        connections.connect(alias=self.alias, host=self.host, port=self.port)

    def disconnect(self):
        connections.disconnect(alias=self.alias)
            
    @staticmethod 
    def __load_collection(host: str, port: str, alias: str, collection_str: str) -> Collection:
        connections.connect(alias=alias, host=host, port=port)
        if utility.has_collection(collection_str, using=alias):
            collection = Collection(collection_str, using=alias)
        else:
            raise ValueError(f"attempted to load orphaned collection {collection_str}, collection no longer exists")
        return collection

    @staticmethod
    def create(host: str, port: int, alias: str, collection_str: str, dim: int) -> VectorDB:
        connections.connect(alias, host=host, port=port)
        if utility.has_collection(collection_str, using=alias):
            raise ValueError(f"collection {collection_str} already exists, cannot build vector database")

        # build vector database schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=dim)
        ]
        schema = CollectionSchema(fields, "mesh vector database store")
        collection = Collection(collection_str, schema, using=alias)
    
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 128},
        }
        collection.create_index("embeddings", index_params=index_params)
        return MilvusDB(host, port, alias, collection_str, collection=collection)

    def add(self, ids: Union[np.ndarray, List], vecs: Union[np.ndarray, List]):
        if isinstance(ids, np.ndarray):
            ids = ids.tolist()
        if isinstance(vecs, np.ndarray):
            vecs = vecs.tolist()

        self.collection.insert([ids, vecs])
        self.collection.flush()
        self.collection.load()

    def delete(self, targets: Union[np.ndarray, List]):
        expr = f"id in {targets}"
        self.collection.delete(expr)
        self.collection.load()

    def erase(self):
        if not utility.has_collection(self.collection_str, using=self.alias):
            raise ValueError(f"attempted to delete non-existent collection {self.colleciton_str}")
        utility.drop_collection(self.collection_str, using=self.alias)

    def search(self, vec: np.ndarray, k: int, search_params: Optional[Dict] = None) -> List[int]:
        if search_params is None:
            search_params = {
                "metric_type": "L2",
                "params": {"nprobe": 10},
            }

        results = self.collection.search(vec.tolist(), "embeddings", search_params, limit=k, output_fields=["id"])
        hits = [list(res.ids) for res in results]
        return list(reduce(lambda a, b: a + b, hits, []))

    def to_json(self, include_instance: bool = False):
        db_info = {
            "host": self.host,
            "port": self.port,
            "alias": self.alias,
            "collection_str": self.collection_str
        }
        result = {
            "db_info": db_info,
            "db_type": "milvus",
        }

        if include_instance:
            result['db'] = self
        return result

    @staticmethod
    def from_json(dict: Dict, return_dict: bool = False):
        try:
            assert dict['db_type'] == "milvus", f"attempted to read non-Milvus database {dict['db_type']} with MilvusDB"

            host = dict['db_info']['host']
            port = dict['db_info']['port']
            alias = dict['db_info']['alias']
            collection_str = dict['db_info']['collection_str']

            db = MilvusDB(host, port, alias, collection_str)

            if return_dict:
                dict['db'] = db
                return dict
            return db 

        except Exception as e:
            raise ValueError(f"error parsing JSON representation: {e}")
