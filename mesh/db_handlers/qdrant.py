import os
import numpy as np

from .base import *
from typing import Optional, Dict
from functools import reduce
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, PointIdsList, SearchRequest, Filter, FieldCondition, MatchValue
from qdrant_client.http.models import Distance, VectorParams

class QdrantDB(VectorDB):
    def __init__(self, path: str, collection_str: str, api_key: Optional[str] = "", client: Optional[QdrantClient] = None):
        self.path = path 
        self.collection_str = collection_str
        self.api_key = api_key
        if client is not None:
            self.client = client
        else:
            self.client = QdrantDB.get_client(path, api_key)
        
    @staticmethod
    def get_client(path: str, api_key: Optional[str] = "") -> QdrantClient:
        if "http" in path:
            client = QdrantClient(url=path, api_key=api_key)
        elif os.path.exists(os.path.dirname(path)):
            client = QdrantClient(path=path, api_key=api_key)
        else:
            raise ValueError(f"unrecognized path {path}")
        return client
                
    @staticmethod
    def create(db_path: str, dim: int, collection_str: str = "data", api_key: Optional[str] = "") -> VectorDB:
        client = QdrantDB.get_client(db_path, api_key=api_key)
        # TODO: Figure out how to get the system to check for whether the collection exists before pulling it
        client.recreate_collection(
            collection_name=collection_str,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        return QdrantDB(db_path, collection_str, api_key=api_key, client=client)

    def exists(self, id: int) -> bool:
        res = self.client.scroll(
            collection_name=self.collection_str,
            scroll_filter=Filter(
                should=[
                    FieldCondition(
                        key="id",
                        match=MatchValue(value=id)
                    )
                ]
            )
        )
        return len(res[0]) > 0

    def add(self, ids: Union[np.ndarray, List], vecs: Union[np.ndarray, List]):
        if isinstance(ids, np.ndarray):
            ids = ids.tolist()
        if isinstance(vecs, np.ndarray):
            vecs = vecs.tolist()

        structs = [
            PointStruct(
                id=idx,
                vector=vec,
                payload={"id": idx}
            ) for (idx, vec) in zip(ids, vecs)
        ]
        self.client.upsert(collection_name=self.collection_str, points=structs)
        
    def delete(self, targets: Union[np.ndarray, List]):
        if not isinstance(targets, List):
            targets = targets.tolist()
        self.client.delete(
            collection_name=self.collection_str,
            points_selector=PointIdsList(
                points=targets,
            ),
        )

    def erase(self):
        # TODO: add check to make sure collection actually exists
        self.client.delete_collection(self.collection_str)

    def search(self, vec: np.ndarray, k: int, **kwargs) -> List[int]:
        assert len(vec.shape) == 2, f"vector search only supports 2d vector input, found shape {vec.shape}"
        requests = [
            SearchRequest(
                vector=v.tolist(),
                limit=k
            ) for v in vec
        ]
        result = self.client.search_batch(
            collection_name=self.collection_str,
            requests=requests
        )
        flat_requests = [r for req in result for r in req]
        return [f.id for f in flat_requests]

    def to_json(self, include_instance: bool = False):
        db_info = {
            "path": self.path,
            "collection_str": self.collection_str,
            "api_key": self.api_key
        }
        result = {
            "db_info": db_info,
            "db_type": "qdrant",
        }

        if include_instance:
            result['db'] = self
        return result

    @staticmethod
    def from_json(dict: Dict, return_dict: bool = False):
        try:
            assert dict['db_type'] == "qdrant", f"attempted to read non-Qdrant database {dict['db_type']} with QdrantDB"

            db = QdrantDB(**dict['db_info'])

            if return_dict:
                dict['db'] = db
                return dict
            return db 

        except Exception as e:
            raise ValueError(f"error parsing JSON representation: {e}")
