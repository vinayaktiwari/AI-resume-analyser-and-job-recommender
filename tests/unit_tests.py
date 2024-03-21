

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
index = pc.Index(index_name)




def test_query(query_text):
    model_name="sentence-transformers/all-mpnet-base-v2"
    model = SentenceTransformer(model_name)

    print("start")
    embeddings = model.encode(sentences=query_text)
    res = index.query(vector=embeddings.tolist(),top_k=2,include_metadata=True)
    print("end")
    print(res)


test_query("Product discussions with stakeholders to finalize on UI functional requirements")