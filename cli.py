import os
import typer

from mesh import key_handler, network_cli
from dotenv import load_dotenv
from typing import Optional

api_key = key_handler.get_key("COHERE_API_KEY")
app = typer.Typer()

@app.command()
def init(host: str, port: int, alias: str, collection_str: str):
    network_cli.init(host, port, alias, collection_str)

@app.command()
def add(file_path: str):
    print(f"API KEY: {api_key}")
    print(f'added filepath {file_path}')

@app.command()
def remove(file_path: str):
    print(f'removed filepath {file_path}')

@app.command()
def index():
    print('indexed')

@app.command()
def search(query: str, k: Optional[int] = None):
    print(f"queried {query} with k={k}")

@app.command()
def update():
    print("updated network")

if __name__ == "__main__":
    app()