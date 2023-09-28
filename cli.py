import os
import typer

from mesh import key_handler, network_cli
from functools import reduce
from typing import Optional, List

api_key = key_handler.get_key("COHERE_API_KEY")
app = typer.Typer()

def read_file_content(file_path: str) -> str:
    _, ext = os.path.splitext(file_path)
    wrapped_content = "{content}"
    match ext:
        case ".txt":
            pass
        case ".md":
            wrapped_content = "```markdown\n{content}\n```"
        case ".py":
            wrapped_content = "```python\n{content}\n```"
        case ".java":
            wrapped_content = "```java\n{content}\n```"
        case _:
            raise ValueError(f"filetype not implemented: {ext}")

    with open(file_path, 'r') as f:
        content = f.read()
    wrapped_content = wrapped_content.format(content=content)
    return wrapped_content

@app.command()
def init(host: str, port: int, alias: str, collection_str: str):
    network_cli.init(host, port, alias, collection_str)

@app.command()
def add(file_path: str, related: Optional[List[str]] = []):
    network = network_cli.load()

    wrapped_content = read_file_content(file_path)
    ids = reduce(lambda a, b: a + b, [network_cli.find_id(network, r) for r in related], [])

    network = network_cli.add(network, wrapped_content, ids, file_path)
    network_cli.close(network)

@app.command()
def push(depth: Optional[int] = 0, verbose: Optional[bool] = False):
    network = network_cli.load()
    network = network_cli.index(network, depth, verbose=verbose)
    network_cli.close(network)

@app.command()
def remove(file_path: str):
    network = network_cli.load()
    network = network_cli.remove(network, file_path)
    network_cli.close(network)

@app.command()
def search(query: str, k: Optional[int] = None):
    network = network_cli.load()
    results = network_cli.search(network, query, k)
    if len(results) == 0:
        print("No Search Results Found")
        return
    print("Search Results:")
    for result in results:
        print("-----------------------------------------------------")
        print(f"Document Filepath: {result['file_path']}")
        print(f"Document Summary:\n{result['summary']}")
        print(f"Document Content:\n{result['content']}")

@app.command()
def synthesize(question: str, num_sources: Optional[int] = None, use_web: bool = False):
    network = network_cli.load()
    response = network_cli.synthesize(network, question, limit=num_sources, use_web=use_web)
    print("Mesh Response: ")
    print(response)
    network_cli.close(network)

if __name__ == "__main__":
    app()