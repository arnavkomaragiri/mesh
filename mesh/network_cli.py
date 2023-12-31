import os

from .network import *
from .key_handlers import *

MESHFILE_PATH = ".mesh/mesh.json"

def init(db_type: str, **kwargs):
    api_key = key_handler.get_key("COHERE_API_KEY")
    network = init_network(api_key, db_type, MESHFILE_PATH, 1024, **kwargs)
    close_network(network, MESHFILE_PATH)

def load() -> Network:
    if not os.path.exists(MESHFILE_PATH):
        raise ValueError(f"unable to find mesh.json file in {os.getcwd()}/.mesh")

    api_key = key_handler.get_key("COHERE_API_KEY")
    return open_network(api_key, MESHFILE_PATH)

def add(network: Network, content: str, related: List[Any] = [], file_path: str = "") -> Network:
    new_id = num_entities(network)

    # find existing entries in the DB that share our filepath
    existing_ids = find_id(network, file_path)
    existing_content = set([network['graph'].nodes[i]['content'] for i in existing_ids])

    # if we have matching content but there are changes, apply the changes and push
    if len(existing_content) > 0 and content not in existing_content:
        network = remove(network, file_path)
        network = add_entity(network, existing_ids[0], content, related, file_path) 
    elif content not in existing_content:
        network = add_entity(network, new_id, content, related, file_path)
    return network

def index(network: Network, depth: int = 0, verbose: bool = False) -> Network:
    return index_network(network, depth, verbose=verbose)

def search(network: Network, query: str, limit: Optional[int] = None, q: Optional[int] = None) -> Tuple[Network, List[Dict]]:
    if limit == None:
        limit = 3

    update_edges = q is not None
    if not update_edges:
        q = 0

    return search_network(network, query, limit=limit, update_edges=update_edges, q=q)

def remove(network: Network, file_path: str):
    ids = find_id(network, file_path)
    for i in ids:
        network = remove_entity(network, i)
    return network

def erase(network: Network):
    delete_network(network)

def synthesize(network: Network, question: str, limit: Optional[int] = None, use_web: bool = False, q: Optional[int] = None) -> Tuple[Network, str]:
    if limit == None:
        limit = 3

    update_edges = q is not None
    if not update_edges:
        q = 0

    network, response = run_synthesis(network, limit, question, use_web=use_web, update_edges=update_edges, q=q)
    return network, response.text

def find_id(network: Network, file_path: str) -> List[Any]:
    return [i for i, d in network['graph'].nodes(data=True) if d['file_path'] == file_path]

def close(network: Network):
    close_network(network, MESHFILE_PATH) 
