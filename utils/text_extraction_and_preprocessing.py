from PyPDF2 import PdfReader
import nltk 
from nltk.corpus import stopwords
from langchain.text_splitter import RecursiveCharacterTextSplitter
nltk.download('stopwords')

def get_pdf_text(pdf_docs)->str: 
    """
    Function takes path of the pdf file and extract text from it.
    """
    text = ""
    pdf_reader = PdfReader(pdf_docs)
    print(len(pdf_reader.pages))  
    for page in pdf_reader.pages:
        text += page.extract_text()

    return text

def remove_stop_words(text):
    """
    Function takes string and removes all the stop words from it and returns a cleaned text.
    """

    nltk.download('stopwords')
    # nlp = spacy.load('en_core_web_sm')
    stop_words = set(stopwords.words("english"))
    words = nltk.word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return " ".join(filtered_words)


def text_splitting(text,chunk_size=2500,chunk_overlap=50):
    """
    Function take a text extracted from pdf and splits it into chunks of size and overlap provided.
    Returns a list of documents of chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.create_documents([text])
    # print(len(docs))
    return chunks