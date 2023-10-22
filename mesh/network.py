import os
import cohere
import json

import numpy as np
import networkx as nx

from tqdm import tqdm

from mesh.db_handlers import *
from mesh.model_handlers.cohere_handler import *
from typing import Union, List, Dict

Network = Dict[str, Union[Dict, np.array, nx.Graph]]

eps = 0.01

def get_entity_summary(client: cohere.Client, content: str, related: List[Dict], model: str, temperature: float) -> str:
    related_summaries = [entity['summary'] for entity in related]
    summarize_prompt = build_summarize_prompt(content, related_summaries) 
    if len(summarize_prompt) < 250:
        return summarize_prompt
    return query_summary(client, model, temperature, document=content, context=related_summaries)

def init_network(api_key: str, db_type: str, path: str, dim: int, **kwargs) -> Network:
    db_dict = db_factory(db_type).create(dim=dim, **kwargs).to_json(include_instance=True)
    
    network = {
        "db": db_dict,
        "path": os.path.abspath(path),
        "client": init_client(api_key),
        "graph": nx.DiGraph(),
        "id_map": {},
        "deleted": [],
    }
    return network

def open_network(api_key: str, path: str) -> Network:
    # load data in from JSON
    with open(path, 'r') as f:
        network = json.load(f)

    # open cohere api/database
    network['client'] = init_client(api_key)
    network['db'] = db_factory(network['db']['db_type']).from_json(network['db'], return_dict=True)

    # rebuild local graph from node link data
    network['graph'] = nx.node_link_graph(network['graph'])
    for n in network['graph'].nodes:
        if 'vec' not in network['graph'].nodes[n]:
            continue
        network['graph'].nodes[n]['vec'] = np.array(network['graph'].nodes[n]['vec'])

    return network

def delete_network(network: Network):
    network['db']['db'].erase()
    if not os.path.exists(network['path']) or not os.path.isfile(network['path']):
        raise ValueError(f"internal network path {network['path']} does not exist, corrupted file")
    os.remove(network['path'])

def close_network(network: Network, out_path: str):
    # close db/cohere connection
    network['db']['db'].disconnect()

    del network['client']
    network['db'] = network['db']['db'].to_json(include_instance=False)

    # convert all numpy arrays to lists for JSON encoding
    for n in network['graph'].nodes:
        if 'vec' not in network['graph'].nodes[n]:
            continue
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
    network['graph'].nodes[node_id]['count'] = 1

    # find connected nodes that are confirmed in the graph
    # TODO: figure out how to handle this, this behavior could silent error if no nodes specified are in graph
    # could throw a warning or something?
    connected_nodes = [network['graph'].nodes[c] for c in connected if c in network['graph']]
    if len(connected_nodes) > 0:
        # build edges for every new connection
        for node in connected_nodes:
            network['graph'].add_edge(node_id, node['id'], weight=1, dynamic=False)
            network['graph'].add_edge(node['id'], node_id, weight=1, dynamic=False)

        # TODO: build a settings type for LLM hyperparams (max tokens, temperature, top-p)
        summary = get_entity_summary(network['client'], content, connected_nodes, 'command', 0.3)

        network['graph'].nodes[node_id]['summary'] = summary
        for c in connected_nodes:
            network['graph'].nodes[c]['update'] = True

    network['graph'].nodes[node_id]['update'] = True
    return network

def num_entities(network: Network) -> int:
    return len(network['graph'].nodes)

def add_edge(network: Network, a: Union[int, str], b: Union[int, str]) -> Network:
    if a not in network['graph'].nodes:
        raise ValueError(f"node {a} not in network")
    if b not in network['graph'].nodes:
        raise ValueError(f"node {b} not in network")
    network['graph'].add_edge(a, b, weight=1, dynamic=False)
    network['graph'].add_edge(b, a, weight=1, dynamic=False)
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

def get_edge_weight(network: Network, source: Any, end: Any) -> float:
    if not network['graph'][source][end]['dynamic']:
        return network['graph'][source][end]['weight']
    node_count = network['graph'].nodes[source]['count']
    edge_count = network['graph'][source][end]['count']
    return edge_count / node_count

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

            # get edge weights in dynamic edges
            weights = [((node_id, i), get_edge_weight(network, node_id, i)) for i in nx.neighbors(network['graph'], node_id)]

            # trim unused dynamic edges for sparsity
            for (n, leaf_id), weight in weights:
                if network['graph'][n][leaf_id]['dynamic'] and weight < eps:
                    network['graph'].remove_edge(n, leaf_id)

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

    replaced_ids = [i for i in ids if network['db']['db'].exists(ids[0])]
    network['db']['db'].delete(replaced_ids + network['deleted'])
    if len(ids) > 0:
        network['db']['db'].add(ids, vectors)
    network['deleted'] = []
    return network

def rewire(network: Network, ret_ids: List[Any], ret_scores: List[float], q: int = 3) -> Network:
    # get best results from search
    idx = np.argsort(ret_scores)[::-1][:q + 1]
    # find top-q pairs to create associations between
    top_id = ret_ids[idx[0]]
    pairs = [(ret_ids[i], top_id) for i in idx[1:]]
    # either introduce new connections or reinforce existing ones for each pair
    for leaf, source in pairs:
        node_count = network['graph'].nodes[leaf]['count']
        if network['graph'].has_edge(leaf, source) and network['graph'][leaf][source]['dynamic']:
            edge_count = network['graph'][leaf][source]['count']

            network['graph'][leaf][source]['count'] = edge_count + 1
            network['graph'].nodes[leaf]['count'] = node_count + 1
            print(f"amplified {leaf}->{source} edge by 1")
        else:
            network['graph'].add_edge(leaf, source, count=1, dynamic=True)
    return network

def search_network(network: Network, nl_query: Union[str, List[str]], limit: int, search_params: Optional[Dict] = None, 
                   update_edges: bool = False, q: int = 3) -> Tuple[Network, List[Dict]]:
    vec = query_embed(network['client'], 'embed-english-light-v2.0', nl_query)
    results, scores = network['db']['db'].search(vec, limit, search_params=search_params)

    idxs = [i for i, hit in enumerate(results) if hit not in network['deleted']]
    ids = [network['id_map'][str(hit)] for hit in results if hit not in network['deleted']]
    scores = [scores[i] for i in idxs] 

    if update_edges:
        network = rewire(network, ids, scores, q=q)

    # TODO: this is sloppy, fix this
    out = [network['graph'].nodes[i] for i in ids]
    for i, res_id in enumerate(ids):
        out[i]['id'] = res_id
    return network, out

def run_synthesis(network: Network, limit: int, question: str, use_web: bool = False, update_edges: bool = False, q: int = 0) -> Tuple[Network, cohere.responses.Chat]:
    queries = get_search_queries(network['client'], 'command', question)
    queries = [q['text'] for q in queries]
    if len(queries) == 0:
        queries = [question]

    network, results = search_network(network, queries, limit, update_edges=update_edges, q=q)
    raw_documents = [{"id": str(r["id"]), "title": r["summary"], "snippet": r["content"], "url": r["file_path"]} for r in results]
    ids = [d['id'] for d in raw_documents]
    _, uidxs = np.unique(ids, return_index=True)
    documents = [raw_documents[i] for i in uidxs]
    response = query_chat(network['client'], 'command', question, sources=documents, use_web=use_web)
    return network, response
