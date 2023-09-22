import json
import cohere
import numpy as np

from functools import reduce
from typing import List, Dict, Tuple, Any, Optional, Union

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

def build_chat_prompt(chatbot_context: str, rules: List[str], prompt: str, example: Optional[str] = None) -> str:
    output_str = ""
    if example is not None:
        output_str = "{chatbot_context}\nRules:\n{rule_str}\nExample Output:\n{example}".format(
            chatbot_context=chatbot_context,
            rule_str=reduce(lambda a, b: f'{a}\n{b}', rules),
            example=example
        )
    else:
        output_str = "{chatbot_context}\nRules:\n{rule_str}\nExample Output:\n{example}".format(
            chatbot_context=chatbot_context,
            rule_str=reduce(lambda a, b: f'{a}\n{b}', rules)
        )
    return output_str + f"\nUser Input: {prompt}\nResponse:"

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

def query_chat(client: cohere.Client, model: str, question: str, chatbot_context: str, rules: List[str], example: Optional[str] = None) -> Dict[str, str]:
    # build chat-specific prompt and run inference
    chat_prompt = build_chat_prompt(chatbot_context, rules=[f"- {rule}" for rule in rules], prompt=question, example=example)
    response = client.chat( 
        model=model,
        message=chat_prompt,
        temperature=0.3,
    ) 

    # parse JSON response and return output
    try:
        output = json.loads(response.text)
    except Exception as e:
        raise ValueError(f"reached error when parsing JSON response:\nResponse:\n{response.txt}\nError:\n{e}")
    return output

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
