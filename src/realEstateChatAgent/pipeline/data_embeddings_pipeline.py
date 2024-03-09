from langchain.llms import GooglePalm
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

if index_name not in pc.list_indexes():
    pc.create_index(spec= {'pod': {'environment': 'gcp-starter',
                  'pod_type': 'starter',
                  'pods': 1,
                  'replicas': 1,
                  'shards': 1}},name=index_name,dimension=768,metric="cosine")
    print("INDEX CREATED SUCCESSFULLY ===========!!!!!!!!!!!!!!!!!!!")
else:
    print("======= Index already exists =========",pc.describe_index(index_name))



index = pc.Index(index_name)

# Define a function to preprocess text
def preprocess_text(text):
    # Replace consecutive spaces, newlines and tabs
    text = re.sub(r'\s+', ' ', text)
    return text



def process_pdf(file_path):
    # create a loader
    loader = PyPDFLoader(file_path)
    # load your data
    data = loader.load()
    # Split your data up into smaller documents with Chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents(data)
    # Convert Document objects into strings
    texts = [str(doc) for doc in documents]
    return texts



llm = GooglePalm(google_api_key=os.environ["GOOGLE_PALM_API_KEY"], temperature=0.1)

# Define a function to create embeddings
def create_embeddings(texts):
    model_name="sentence-transformers/all-mpnet-base-v2"
    model = SentenceTransformer(model_name)

    print("start")
    embeddings = model.encode(sentences=texts)

    print("end")
    print(embeddings)
    return embeddings



# Define a function to upsert embeddings to Pinecone
def upsert_embeddings_to_pinecone(index, embeddings, ids):
    index.upsert(vectors=[(str(id), embedding) for id, embedding in zip(ids, embeddings)])

# Process a PDF and create embeddings
file_path = "/home/vinayak.t/Real-Estate-Chat-Agent/data/b1-1.pdf"  # Replace with your actual file path
texts = process_pdf(file_path)
id_list = [index for index,_ in enumerate(texts)]


print("text extracted")
embeddings = create_embeddings(texts)

# Upsert the embeddings to Pinecone
upsert_embeddings_to_pinecone(index, embeddings, id_list)
print("upserted")

