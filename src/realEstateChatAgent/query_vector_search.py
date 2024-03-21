from langchain.llms import GooglePalm
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import Pinecone
from sentence_transformers import SentenceTransformer
import os

# Load the SentenceTransformer model for encoding text into embeddings
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# Define the function for similarity search
def similarity_search(query_embeddings):
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    index_name = os.environ['INDEX_NAME']

    index = pc.Index(index_name)
    response = index.query(vector=query_embeddings, top_k=2,include_values=False)
    return response['matches']


# Define the function to retrieve relevant documents based on similarity search results
def retrieve_relevant_documents(similarity_results):
    relevant_documents = []
    for match in similarity_results:
        if match['score'] >= 0.2:  # Adjust the score threshold as needed
            relevant_documents.append(match['text'])
    print("RELEVANT DOCUMENTS =====================" ,relevant_documents)
    return relevant_documents

# Define the function to get response from Google Palm LLM
def get_llm_response(query_text, relevant_documents):
    llm = GooglePalm(google_api_key=os.environ['GOOGLE_PALM_API_KEY'], temperature=0.1)  # Adjust temperature as needed
    prompt = """You are a helpful and informative bot that answers questions using text from the reference passage included below. \
    Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. \
    However, you are talking to a non-technical audience, so be sure to break down complicated concepts and \
    strike a friendly and converstional tone. \
    If the passage is irrelevant to the answer, you may ignore it.
    QUESTION: '{query_text}'
    PASSAGE: '{relevant_documents}'"""
    prompt = f"Query: {query_text}\nContext: {relevant_documents}"

    response=llm.generate(prompts=[prompt])

    return response

  # Format relevant documents properly
    # input_documents = [{"text_" + str(i+1):doc} for i,doc in enumerate(relevant_documents)]

    # print("input_doc",input_documents)
    # chain = load_qa_chain(llm, chain_type="stuff")
    # response = chain.run(question=query_text, input_documents=re)
# Function to process user query
def process_user_query(query_text):
    # Convert query text to embeddings
    query_embeddings = model.encode(sentences=query_text)
    print("Converted query text to embeddings")

    # Perform similarity search
    similarity_results = similarity_search(query_embeddings.tolist())
    print("Performed similarity search")

    # Retrieve relevant documents
    relevant_documents = retrieve_relevant_documents(similarity_results)
    print("Retrieved relevant documents")
    
    # Generate prompt with context
    # context_text = " ".join([str(doc) for doc in context_text])
    # print("================CONTEXT TEXT ",  context_text)
    # Get response from LLM
    response = get_llm_response(query_text,relevant_documents=relevant_documents)
    
    return response

# Test the function with user query
user_query ="who is manmeet kaur"
response = process_user_query(user_query)
print("LLM Response:", response)
