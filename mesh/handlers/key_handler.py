import os

from dotenv import load_dotenv, find_dotenv

if os.path.exists(".mesh/keys.env"):
    load_dotenv(".mesh/keys.env")
else:
    raise ValueError(f"missing expected keys.env file in {os.getcwd()}/.mesh/")

def get_key(varname: str) -> str:
    return os.getenv(varname)