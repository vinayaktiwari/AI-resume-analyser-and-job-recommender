from llama_index import VectorStoreIndex
from langchain import LangChainModel
from pymongo import MongoClient
import torch

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['your_database']
collection = db['your_collection']

# Sample text data retrieval from MongoDB
data = []
for doc in collection.find({}, {"_id": 0, "text_field": 1}):  # Assuming "text_field" is the field containing text data
    data.append(doc["text_field"])

# Load the LangChain model
model = LangChainModel.from_pretrained("all-mpnet-base-v2")

# Generate embeddings for the text data
embeddings = []
for text in data:
    embedding = model.encode(text)
    embeddings.append(embedding)

# Convert embeddings to numpy array
embeddings = torch.cat(embeddings).numpy()

# Initialize VectorStoreIndex
index = VectorStoreIndex()

# Index data and embeddings
for idx, doc in enumerate(data):
    index.add_item(doc, embeddings[idx])

# Save the index
index.save("your_vectorstore_index.index")

# Close the MongoDB connection
client.close()
