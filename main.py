from ragserver import rag_application

# Initialize the RAG application
# Example usage
question = "How old is Ali Gharbi ?"
answer = rag_application.run(question)
print("Question:", question)
print("Answer:", answer)
