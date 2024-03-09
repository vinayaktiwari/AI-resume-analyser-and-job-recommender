from langchain.llms import GooglePalm
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import os
import re
import pinecone
from dotenv import load_dotenv
from datasets import load_dataset
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
load_dotenv()



index_name = os.environ['INDEX_NAME']
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
# pinecone.init(os.environ['PINECONE_API_KEY'],os.environ['PINECONE_ENV'])
index = pc.Index(index_name)


#Vector Search

def vector_search(query):
    model_name="sentence-transformers/all-mpnet-base-v2"
    model = SentenceTransformer(model_name)
    Rag_data = ""
    embeddings = model.encode(sentences=query)
    response = index.query(vector=embeddings.tolist(),top_k=2,include_metadata=True)
    for match in response['matches']:
        if  match['score']<0.80:
            continue
        Rag_data += match['metadata']['text']
    
    return Rag_data


def make_prompt(query, )




llm = GooglePalm(google_api_key=os.environ['GOOGLE_PALM_API_KEY'],temperature=0.1)

def palm_llm_response(prompt,rag):









llm.generate(prompts=)