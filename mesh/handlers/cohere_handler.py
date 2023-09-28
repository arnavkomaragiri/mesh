import json
import cohere
import numpy as np

from functools import reduce
from typing import List, Dict, Tuple, Any, Optional, Union

def get_search_queries(client: cohere.Client, model: str, question: str) -> List[str]:
    response = query_chat(client, model, question, search_queries_only=True)
    return response.search_queries

# instantiates client object, this will be passed around to every query system
def init_client(api_key: str, **kwargs) -> cohere.Client:
    return cohere.Client(api_key, **kwargs)

# builds chain of thought prompt factoring in rules and context
def build_cot_prompt(prompt: str, rules: List[str], context_str: Optional[str] = None, output_name: str = "Response") -> str:
    if context_str is not None:
        return "Rules:\n{rule_str}\nContext:\n{context_str}\nTask: {prompt}\n{output}: ".format(prompt=prompt, 
                                                                                    context_str=context_str, 
                                                                                    rule_str=reduce(lambda a, b: f'{a}\n{b}', rules),
                                                                                    output=output_name)
    return "Rules:\n{rule_str}\nTask: {prompt}\n{output}: ".format(prompt=prompt, rule_str=reduce(lambda a, b: f'{a}\n{b}', rules),
                                                                   output=output_name)

def build_summarize_prompt(content: str, context: List[str] = []) -> str:
    summarize_prompt = ""
    if len(context) > 0:
        context = [f"- {doc}" for doc in context]
        summarize_prompt = "Context: {context_str}\n".format(context_str=reduce(lambda a, b: "{a}\n{b}", context))
    summarize_prompt += f"Content:\n{content}" 
    return summarize_prompt

def query_generate(client: cohere.Client, model: str, num_tokens: int, temperature: float, p: float,
                   prompt: str, rules: List[str], context_str: str, output_name: str = "Response",
                   **kwargs) -> str:
    # build chain-of-thought prompt for generation with rules and context
    cot_prompt = build_cot_prompt(prompt, rules=[f"- {rule}" for rule in rules], context_str=context_str, output_name=output_name)
    # run inference
    response = client.generate(
        model=model,
        prompt=cot_prompt,
        max_tokens=num_tokens,
        temperature=temperature,
        p=p,
        **kwargs
    )
    # TODO: potentially integrate a reward model to do ranking of results from multiple traces
    return response.generations[0].text

def query_summary(client: cohere.Client, model: str, temperature: float, 
                  document: Optional[str] = None, context: List[str] = [], summarize_prompt: Optional[str] = None,
                  additional_command: str = "Be as brief as possible") -> str:
    if document is None and summarize_prompt is None:
        raise ValueError("must provide either document or prebuilt summarize prompt to run summarization")
    elif summarize_prompt is None:
        summarize_prompt = build_summarize_prompt(document, context)

    response = client.summarize(
        text=summarize_prompt,
        model=model,
        format='auto',
        length='auto',
        additional_command=additional_command,
        temperature=temperature
    )
    return response.summary

def query_chat(client: cohere.Client, model: str, question: str, sources: List[Dict[str, str]] = [], 
               search_queries_only: bool = False, use_web: bool = False) -> cohere.responses.Chat:
    if use_web:
        response = client.chat( 
            model=model,
            message=question,
            temperature=0.3,
            prompt_truncation="auto",
            search_queries_only=search_queries_only,
            connectors=[{"id": "web-search"}]
        )
    else:
        response = client.chat( 
            model=model,
            message=question,
            temperature=0.3,
            prompt_truncation="auto",
            documents=sources,
            search_queries_only=search_queries_only,
            connectors = []
        ) 
    return response
    
def query_embed(client: cohere.Client, model: str, text: Union[str, List[str]]) -> np.array:
    # if we're embedding a single item, wrap it in a list
    if isinstance(text, str):
        text = [text]

    # run embed inference and wrap in numpy array
    response = client.embed(
        model=model,
        texts=text
    )
    return np.array(response.embeddings)
