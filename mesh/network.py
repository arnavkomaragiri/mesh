import os
import cohere
import json

import numpy as np
import networkx as nx

from tqdm import tqdm
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

from mesh.handlers.cohere_handler import *
from functools import reduce
from typing import Union, List, Dict

Network = Dict[str, Union[Dict, np.array, nx.Graph]]

def get_entity_summary(client: cohere.Client, content: str, related: List[Dict], model: str, temperature: float) -> str:
    related_summaries = [entity['summary'] for entity in related]
    summarize_prompt = build_summarize_prompt(content, related_summaries) 
    if len(summarize_prompt) < 250:
        return summarize_prompt
    return query_summary(client, model, content, temperature, related_summaries)

def init_network(api_key: str, host: str, port: int, alias: str, dim: int, collection_str: str = "data") -> Network:
    # connect to milvus and check if we're making a new collection
    connections.connect(alias, host=host, port=port)
    if utility.has_collection(collection_str, using=alias):
        raise ValueError(f"collection {collection_str} already exists, cannot build network")

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

    network = {
        "milvus_info": {
            "host": host,
            "port": port,
            "alias": alias,
            "collection_str": collection_str, 
            "collection": collection,
        },
        "client": init_client(api_key),
        "graph": nx.Graph(),
        "id_map": {},
        "deleted": [],
    }
    # load collection
    network['milvus_info']['collection'].load()
    return network

def open_network(api_key: str, path: str) -> Network:
    # load data in from JSON
    with open(path, 'r') as f:
        network = json.load(f)
    
    # open cohere/milvus api 
    network['client'] = init_client(api_key)
    milvus_info = network['milvus_info']
    connections.connect(alias=milvus_info['alias'], host=milvus_info['host'], port=milvus_info['port']) 
    if utility.has_collection(milvus_info['collection_str']):
        network['milvus_info']['collection'] = Collection(milvus_info['collection_str'])
    else:
        raise ValueError(f"attempted to load orphaned collection {milvus_info['collection_str']}, collection no longer exists")

    # rebuild local graph from node link data
    network['graph'] = nx.node_link_graph(network['graph'])
    for n in network['graph'].nodes:
        network['graph'].nodes[n]['vec'] = np.array(network['graph'].nodes[n]['vec'])

    return network

def delete_network(network: Network):
    collection_name = network['milvus_info']['collection_str']
    if not utility.has_collection(collection_name, using=network['milvus_info']['alias']):
        raise ValueError(f"attempted to delete non-existent collection {collection_name}")
    utility.drop_collection(collection_name, using=network['milvus_info']['alias'])

def close_network(network: Network, out_path: str):
    # close milvus/cohere connection
    connections.disconnect(network['milvus_info']['alias'])
    del network['client']
    del network['milvus_info']['collection']

    # convert all numpy arrays to lists for JSON encoding
    for n in network['graph'].nodes:
        if isinstance(network['graph'].nodes[n]['vec'], np.ndarray):
            network['graph'].nodes[n]['vec'] = network['graph'].nodes[n]['vec'].tolist()

    # encode graph data into JSON structure
    link_data = nx.node_link_data(network['graph'])
    network['graph'] = link_data

    # write to file 
    with open(out_path, 'w') as f:
        json.dump(network, f)

def add_entity(network: Network, node_id: Union[str, int], content: str, connected: List[Any] = [],
               file_path: Optional[str] = "") -> Network:
    if node_id in network['graph'].nodes:
        raise ValueError(f"entity {node_id} already exists in graph")
    network['graph'].add_node(node_id)

    # set preliminary metadata
    network['graph'].nodes[node_id]['id'] = node_id
    network['graph'].nodes[node_id]['content'] = content
    network['graph'].nodes[node_id]['file_path'] = file_path

    # find connected nodes that are confirmed in the graph
    # TODO: figure out how to handle this, this behavior could silent error if no nodes specified are in graph
    # could throw a warning or something?
    connected_nodes = [network['graph'].nodes[c] for c in connected if c in network['graph']]
    if len(connected_nodes) > 0:
        network['graph'].add_edge(node_id, connected_nodes['id'])
        # TODO: build a settings type for LLM hyperparams (max tokens, temperature, top-p)
        summary = get_entity_summary(network['client'], content, connected_nodes, 'command', 0.3)

        network['graph'].nodes[node_id]['summary'] = summary
        for c in connected_nodes:
            network['graph'].nodes[c]['update'] = True

    network['graph'].nodes[node_id]['update'] = True
    return network

def add_edge(network: Network, a: Union[int, str], b: Union[int, str]) -> Network:
    if a not in network['graph'].nodes:
        raise ValueError(f"node {a} not in network")
    if b not in network['graph'].nodes:
        raise ValueError(f"node {b} not in network")
    network['graph'].add_edge(a, b)
    network['graph'].nodes[a]['update'] = True
    network['graph'].nodes[b]['update'] = True
    return network

def remove_entity(network: Network, node_id: Union[int, str]) -> Network:
    if node_id not in network['graph'].nodes:
        raise ValueError(f"attempted to delete entity {node_id} not in graph")

    del network['id_map'][str(hash(node_id))]
    network['graph'].remove_node(node_id)
    network['deleted'] += [node_id]
    return network

def index_network(network: Network, depth: int, verbose: bool = False) -> Network:
    if depth < 0:
        raise ValueError(f"invalid update depth {depth}, depth must be > 0")
    queue = [[n for n in network['graph'].nodes if network['graph'].nodes[n]['update'] or 'vec' not in network['graph'].nodes[n]], []]
    ids = []
    vectors = []
    visited_set = set()

    iterator = range(depth + 1)
    if verbose:
        iterator = tqdm(iterator, total=len(iterator), unit="levels")

    for _ in iterator:
        for node_id in queue[0]:
            node = network['graph'].nodes[node_id]
            if node_id in visited_set:
                continue

            neighbors = [network['graph'].nodes[i] for i in nx.neighbors(network['graph'], node_id) if i not in visited_set]
            if node['update']:
                summary = get_entity_summary(network['client'], node['content'], neighbors, 'command', 0.3)

                network['graph'].nodes[node_id]['summary'] = summary
                network['graph'].nodes[node_id]['update'] = False

            vec = query_embed(network['client'], 'embed-english-light-v2.0', network['graph'].nodes[node_id]['summary']).squeeze()
            network['graph'].nodes[node_id]['vec'] = vec
            vectors += [vec.tolist()]

            if node_id not in network['id_map'].values():
                # create new entry in id map
                network['id_map'][str(hash(node_id))] = node_id
            ids += [hash(node_id)]
            
            visited_set.add(node_id)
            queue[1] += neighbors
        queue = queue[1:]

    expr = f"id in {ids + network['deleted']}"
    network['milvus_info']['collection'].delete(expr)
    if len(ids) != 0:
        network['milvus_info']['collection'].insert([ids, vectors]) 
        network['milvus_info']['collection'].flush()
    network['milvus_info']['collection'].load()
    network['deleted'] = []
    return network

def search_network(network: Network, nl_query: Union[str, List[str]], limit: int, search_params: Optional[Dict] = None) -> List[Dict]:
    if search_params is None:
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10},
        }
    vec = query_embed(network['client'], 'embed-english-light-v2.0', nl_query).tolist()
    results = network['milvus_info']['collection'].search(vec, "embeddings", search_params, limit=limit, output_fields=["id"])
    hits = [list(res.ids) for res in results]
    ids = [network['id_map'][str(hit)] for hit in reduce(lambda a, b: a + b, hits, [])]

    # TODO: this is sloppy, fix this
    out = [network['graph'].nodes[i] for i in ids]
    for i, res_id in enumerate(ids):
        out[i]['id'] = res_id
    return out

def run_synthesis(network: Network, limit: int, question: str, use_web: bool = False) -> str:
    queries = get_search_queries(network['client'], 'command', question)
    queries = [q['text'] for q in queries]

    results = search_network(network, queries, limit)
    documents = [{"id": str(r["id"]), "title": r["summary"], "snippet": r["content"], "url": r["file_path"]} for r in results]
    response = query_chat(network['client'], 'command', question, sources=documents, use_web=use_web)
    return response
