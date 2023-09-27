import os
import logging

from .network import *
from .handlers import *

MESHFILE_PATH = ".mesh/mesh.json"

def init(host: str, port: int, alias: str, collection_str: str):
    api_key = key_handler.get_key("COHERE_API_KEY")
    network = init_network(api_key, host, port, alias, 1024, collection_str)
    logging.debug("initialized mesh network")
    close_network(network, MESHFILE_PATH)
    logging.debug(f"wrote mesh network file to {os.path.abspath(MESHFILE_PATH)}")
    logging.info("successfully initialized mesh repository")

def load() -> Network:
    if not os.path.exists(MESHFILE_PATH):
        raise ValueError(f"unable to find mesh.json file in {os.getcwd()}/.mesh")

    api_key = key_handler.get_key("COHERE_API_KEY")
    return open_network(api_key, MESHFILE_PATH)

def close(network: Network):
    close_network(network, MESHFILE_PATH) 
