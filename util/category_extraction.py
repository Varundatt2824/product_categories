def category_extraction(llm,product_list,summary):
    
    from langchain.schema import(
            AIMessage,
            HumanMessage,
            SystemMessage
        )
    chat_messages=[
    SystemMessage(content='You are an multilingual expert in understanding regulatory documents.\
                           You are given a list of product and summary of regulatory document.\
                           Given the regulation text from pdf you are suppose to output a list of product categories which are relavant to this text:\
                           Give the output in a with their relavance score and a reason for why they are the best choices \n\
                           The products should be strictly coming from the product list given\
                           The output should be in json format'),
    HumanMessage(content=f'List of products are : {product_list}\n\
                 summary: {summary}')
    ]
    res=llm(chat_messages).content  
    return res
