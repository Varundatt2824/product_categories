def category_extraction(llm,product_list,summary):
    """
    Function takes a LLM model, product list and the summary of extracted text from pdf 
    and outputs the list of product categories which are relavant to text.
    """
    
    from langchain.schema import(
            AIMessage,
            HumanMessage,
            SystemMessage
        )
    chat_messages=[
    SystemMessage(content="""You are an multilingual expert in understanding regulatory documents.
                           You are given a list of product and summary of regulatory document.
                           Given the regulation text from pdf you are suppose to output a list of product categories which are relavant to this text:
                           Give the output in a with their relavance score and an explaination for choosing that category \n
                           The products should be strictly coming from the product list given
                           The output should be in json format
                  An example of output is given below
                        {relavant products:[
                            0:{product:
                                score:
                                reason:
                            },
                            1:{product:
                                score:
                                reason:
                            },
                        ] 
                        } """),
    HumanMessage(content=f'List of products are : {product_list}\n\
                 summary: {summary}')
    ]
    res=llm(chat_messages).content  
    return res
