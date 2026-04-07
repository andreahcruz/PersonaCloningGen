import chromadb
client = chromadb.PersistentClient(path="./chroma_db") # Update to your path
collection = client.get_collection(name="persona_clones")

# This is the "Truth Check"
results = collection.get(include=["metadatas"])
sources = [m.get("source") for m in results["metadatas"]]

from collections import Counter
print("--- DATA DISTRIBUTION IN CHROMADB ---")
print(Counter(sources))