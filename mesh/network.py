import cohere
import json

import numpy as np
import networkx as nx

from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

from cohere_handler import *
from typing import Dict


Network = Dict[str, Union[Dict, np.array, nx.Graph]]

def get_entity_summary(client: cohere.Client, related: List[Dict], model: str, num_tokens: int, temperature: float, **kwargs) -> str:
    prompt = "Generate a brief summary of the specified content while factoring in all of the provided context. Make sure to explain how the provided content connects to the quoted contexts. Only use information provided in the above content, do not generate your own information. Do not repeat sentences or get caught in loops."
    rules = ["Be honest and clear as possible, do not hallucinate information and do not cite information not explicitly mentioned",
             "Be as concise as possible, you are generating text for individuals with a low attention span",
             "Capture the core concepts of the specified content while avoiding unnecessary words",
             "Use the specified context to inform the summarization process; the content given exists within a larger graph structure",
             "The <DOCUMENT> tag corresponds to where the specified content exists within the larger data structure"]
    context_str = reduce(lambda a, b: f'{a}\n{b}', [f"- {entity['summary']}" for entity in related if 'summary' in entity])
    return query_generate(client, model, num_tokens, temperature, 
                          prompt=prompt, rules=rules, context_str=context_str, output_name="Summary", **kwargs)

def init_network(api_key: str, host: str, port: int, alias: str, dim: int, collection_str: str = "data") -> Network:
    # connect to milvus and check if we're making a new collection
    connections.connect(alias, host=host, port=port)
    if utility.has_collection(collection):
        raise ValueError(f"collection {collection_str} already exists, cannot build network")

    # build vector database schema
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=dim)
    ]
    schema = CollectionSchema(fields, "mesh vector database store")
    collection = Collection(collection_str, schema)

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
        "id_map": Dict[int, Union[int, str]]
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
    network['milvus_info']['collection'] = Collection(milvus_info['collection_str'])
    network['milvus_info']['collection'].load()

    # rebuild local graph from node link data
    network['graph'] = nx.node_link_graph(network['graph'])
    return network

def close_network(network: Network, out_path: str):
    # close milvus/cohere connection
    connections.disconnect(network['milvus_info']['alias'])
    del network['client']
    del network['milvus_info']['collection']

    # encode graph data into JSON structure
    link_data = nx.node_link_data(network['graph'])
    network['graph'] = link_data

    # write to file 
    with open(out_path, 'w') as f:
        json.dump(network, f)

def add_entity(network: Network, node_id: Union[str, int], connected: List[Any], content: str, 
               file_path: Optional[str] = None) -> Network:
    if node_id in network['graph']:
        raise ValueError(f"entity {node_id} already exists in graph")
    network['graph'].add_node(node_id)

    # create new entry in id map
    network['id_map'][hash(node_id)] = node_id

    # set preliminary metadata
    network['graph'].nodes[node_id]['id'] = node_id
    network['graph'].nodes[node_id]['content'] = content
    network['graph'].nodes[node_id]['file_path'] = file_path

    # find connected nodes that are confirmed in the graph
    # TODO: figure out how to handle this, this behavior could silent error if no nodes specified are in graph
    # could throw a warning or something?
    connected_nodes = [network['graph'][c] for c in connected if c in network['graph']]
    if len(connected_nodes) > 0:
        network['graph'].add_edge(node_id, connected_nodes['id'])
        # TODO: build a settings type for LLM hyperparams (max tokens, temperature, top-p)
        summary = get_entity_summary(network['client'], connected_nodes, 'command-light', 500, 0.5, p=0.95)
        network['graph'].nodes[node_id]['summary'] = summary
        for c in connected_nodes:
            network['graph'].nodes[c]['update'] = True

    network['graph'].nodes[node_id]['update'] = False
    return network

def index_network(network: Network, depth: int) -> Network:
    queue = [[n for n in network['graph'] if n['modified'] or 'vec' not in n], []]
    ids = []
    vectors = []
    visited_set = set()
    for _ in range(depth):
        for node in queue[0]:
            node_id = node['id']
            if node_id in visited_set:
                continue

            neighbors = [network['graph'][i] for i in nx.neighbors(network['graph'], node_id)]
            if node['update']:
                network['graph'].nodes[node_id]['summary'] = get_entity_summary(network['client'], neighbors, 
                                                                                'command-light', 500, 0.5, p=0.95)
            vec = query_embed(network['client'], 'embed-english-light-v2.0', network['graph'][node_id]['summary'])
            network['graph'].nodes[node_id]['vec'] = vec
            vectors += vec
            ids += [node_id]
            
            visited_set.add(node_id)
            queue[1] += neighbors
        queue = queue[1:]
    
    ids = np.concatenate(ids)
    vectors = np.vstack(vectors)

