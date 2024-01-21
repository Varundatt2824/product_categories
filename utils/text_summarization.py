from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
import tiktoken

def num_tokens_from_string(string: str, encoding_name: str) -> int:   
    """
    Function takes a string and llm model name and return the number of tokens.
    """ 
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    print(num_tokens)
    return num_tokens


def text_summarization(docs,llm)->str:
    """
    Function takes a string a list of documented chunks and the LLM model and 
    summarize those chunks and return as a string.
    """
    
    prompt_template = """Write a concise summary of the following:
    {text}
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["text"]) 
    chain = load_summarize_chain(llm, chain_type="map_reduce", map_prompt=prompt, combine_prompt=prompt, verbose=False)
    summary = chain.invoke(docs)
    return summary['output_text']