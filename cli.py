import os
import typer

from mesh import key_handler, network_cli
from functools import reduce
from typing import Optional, List, Dict, Tuple

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

def get_args(ctx: typer.Context) -> Tuple[List, Dict]:
    args, kwargs = [], {}
    ctx_args, i = ctx.args, 1

    if len(ctx_args) == 0:
        return args, kwargs

    while True:
        prev_arg, curr_arg = ctx_args[i - 1], ctx_args[min(i, len(ctx_args) - 1)]
        is_prev_key, is_curr_key = (prev_arg[:2] == "--"), (curr_arg[:2] == "--")
        match (is_prev_key, is_curr_key):
            case (True, True):
                kwargs[prev_arg[2:]] = True
                i += 1
            case (True, False):
                kwargs[prev_arg[2:]] = curr_arg
                i += 2
            case (False, True):
                args += [prev_arg]
                i += 1
            case (False, False):
                args += [prev_arg, curr_arg]
                i += 2
        if i > len(ctx_args):
            break
    return args, kwargs

@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def init(db_type: str, ctx: typer.Context):
    args, kwargs = get_args(ctx)
    network_cli.init(db_type, *args, **kwargs)

@app.command()
def add(file_path: str, related: Optional[List[str]] = []):
    network = network_cli.load()

    # convert filepath to absolute to avoid collision
    file_path = os.path.abspath(file_path)

    # wrap content and find related ids
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
    file_path = os.path.abspath(file_path)
    network = network_cli.load()
    network = network_cli.remove(network, file_path)
    network_cli.close(network)

@app.command()
def erase():
    response = input("WARNING: This is a permanent operation. Are you sure about this? (y or n): ")
    if response.lower() == "y":
        network = network_cli.load()
        network_cli.erase(network)
    else:
        print("operation refused")

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