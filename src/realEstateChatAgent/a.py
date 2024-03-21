from langchain.llms import GooglePalm
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.chains.question_answering import load_qa_chain

import os
import re
import pinecone
from dotenv import load_dotenv
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
load_dotenv()

index_name = os.environ['INDEX_NAME']
pinecone_api =os.environ['PINECONE_API_KEY']

# initialize pinecone
pinecone.init(
    api_key=pinecone_api,  # find at app.pinecone.io
    environment=os.environ['PINECONE_ENV'] # next to api key in console
)

     

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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    documents = text_splitter.split_documents(data)
    print("DOCUMENTS================",documents)
    print("LENGHT================",len(documents))
    page_content =[t.page_content  for t in documents]
    return documents,page_content



llm = GooglePalm(google_api_key=os.environ["GOOGLE_PALM_API_KEY"], temperature=0.1)

# Define a function to create embeddings
def create_embeddings():
    model="sentence-transformers/all-mpnet-base-v2"
    embeddings = HuggingFaceBgeEmbeddings(model_name=model)
    return embeddings


# # Define a function to upsert embeddings to Pinecone
def upsert_embeddings_to_pinecone(index, embeddings,page_content_list):
    i= Pinecone.from_texts(page_content_list,embedding=embeddings,index_name=index)
    return i

# # Process a PDF and create embeddings
# file_path = "/home/vinayak.t/Real-Estate-Chat-Agent/data/Manmeet_Kaur_Resume.pdf"  # Replace with your actual file path
# chunks,content = process_pdf(file_path)
# print("========================text extracted============================")
# embeddings = create_embeddings()
# print("========================embeddings created============================")
# # # Upsert the embeddings to Pinecone
# index =upsert_embeddings_to_pinecone(index_name, embeddings,content)
# print("================embeddings upserted===============================")







#================== RETRIEVE QUERY ===============================



## Cosine Similarity Retreive Results from VectorDB
def retrieve_query(index_name,query,k=20):
    # matching_results=index.similarity_search(query,k=k)
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    emb = model.encode(query)
    index = pinecone.Index(index_name=index_name)
    res = index.query(emb.tolist(),top_k=k,include_metadata=True)
    print(res['matches'])
    return res['matches']

# # Define the function for similarity search
# def similarity_search(query_embeddings):
#     pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
#     index_name = os.environ['INDEX_NAME']

#     index = pc.Index(index_name)
#     response = index.query(vector=query_embeddings, top_k=2,include_values=False)
#     return response['matches']


# Define the function to retrieve relevant documents based on similarity search results
def retrieve_relevant_documents(similarity_results):
    relevant_documents = []
    for match in similarity_results:
        if match['score'] >= 0.2:  # Adjust the score threshold as needed
            relevant_documents.append(match['metadata']['text'])
    print("RELEVANT DOCUMENTS =====================" ,relevant_documents)
    return relevant_documents




    # return matching_results


llm = GooglePalm(google_api_key=os.environ['GOOGLE_PALM_API_KEY'], temperature=0.1)  # Adjust temperature as needed

# Define the function to get response from Google Palm LLM
def get_llm_response(query_text, relevant_documents):
    llm = GooglePalm(google_api_key=os.environ['GOOGLE_PALM_API_KEY'], temperature=0.1)  # Adjust temperature as needed
    prompt = """You are a helpful and informative bot that answers questions using text from the reference passage included below. 
    Do not answer anything apart from the given context, you will only answer queries regarding the resume of the candidate provided to you

    Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. 
    However, you are talking to a non-technical audience, so be sure to break down complicated concepts and 
    strike a friendly and converstional tone. 

    If the passage is irrelevant to the answer, you may ignore it.
    QUESTION: '{query_text}'
    PASSAGE: '{relevant_documents}'"""
    prompt = f"Query: {query_text}\nContext: {relevant_documents}"
    response=llm.generate(prompts=[prompt])
    return response

query = "How many years of work experience she has?"
similarity_results=retrieve_query(index_name,query)

doc_search = retrieve_relevant_documents(similarity_results)
print("=========================QUERY EMBEDDINGS RETREIVED ===============================================")




answer = get_llm_response(query_text=query,relevant_documents=doc_search)
print(answer)





def input_pdf_text(uploaded_file):
    reader=pdf.PdfReader(uploaded_file)
    text=""
    for page in range(len(reader.pages)):
        page=reader.pages[page]
        text+=str(page.extract_text())
    return text