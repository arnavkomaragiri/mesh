import typer

from typing import Optional

app = typer.Typer()

@app.command()
def add(file_path: str):
    print(f'added filepath {file_path}')

@app.command()
def remove(file_path: str):
    print(f'removed filepath {file_path}')

@app.command()
def search(query: str, k: Optional[int] = None):
    print(f"queried {query} with k={k}")

@app.command()
def update():
    print("updated network")

if __name__ == "__main__":
    app()