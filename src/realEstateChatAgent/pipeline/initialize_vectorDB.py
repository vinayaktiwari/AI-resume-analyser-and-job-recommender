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
load_dotenv()

index_name = os.environ['INDEX_NAME']

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
# pinecone.init(os.environ['PINECONE_API_KEY'],os.environ['PINECONE_ENV'])

# if index_name not in pc.list_indexes():
#     pc.create_index(spec= {'pod': {'environment': 'gcp-starter',
#                   'pod_type': 'starter',
#                   'pods': 1,
#                   'replicas': 1,
#                   'shards': 1}},name=index_name,dimension=3,metric="cosine")
#     print("INDEX CREATED SUCCESSFULLY ===========!!!!!!!!!!!!!!!!!!!")
# else:
#     print("======= Index already exists =========",pc.describe_index(index_name))



# # Sample data
# data = [
#     {"id": "doc3", "values": [0.7, 0.2, 0.3]},  # Replace with your actual vector
#     # Add more data as needed
# ]

# # Load data into Pinecone

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


print("started")
embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

print(embeddings)





# # Define a function to create embeddings
# def create_embeddings(texts):
#     embeddings_list = []
#     for text in texts:
        
#         llm = GooglePalm(google_api_key=os.environ["GOOGLE_PALM_API_KEY"], temperature=0.1)
    
#         embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

#         res = openai.Embedding.create(input=[text], engine=MODEL)
#         embeddings_list.append(res['data'][0]['embedding'])
#     return embeddings_list




# # Define a function to upsert embeddings to Pinecone
# def upsert_embeddings_to_pinecone(index, embeddings, ids):
#     index.upsert(vectors=[(id, embedding) for id, embedding in zip(ids, embeddings)])

# # Process a PDF and create embeddings
# file_path = "your_pdf_here.pdf"  # Replace with your actual file path
# texts = process_pdf(file_path)
# embeddings = create_embeddings(texts)

# # Upsert the embeddings to Pinecone
# upsert_embeddings_to_pinecone(index, embeddings, [file_path])













# index.upsert(data)


# print("Data loaded into Pinecone successfully.!!!!!!!!!!!!!!!!!!!")




#=============================
# index = pc.Index(index_name)
# data = load_dataset('/home/vin-kar/Real-Estate-Chat-Agent/data/b1-1.pdf', split='train')
# print(data)



# llm = GooglePalm(google_api_key=os.environ["GOOGLE_PALM_API_KEY"], temperature=0.1)


# embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


