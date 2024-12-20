from langchain_community import embeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import (ChatOllama, OllamaEmbeddings)
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_community.vectorstores import SKLearnVectorStore

loader = WebBaseLoader(
    web_path = "https://en.wikipedia.org/wiki/Ali_Gharbi",
    header_template = None,
    # verify_ssl = True,
    # proxies = None,
    # continue_on_failure = False,
    # autoset_encoding = True,
    # encoding = None,
    # web_paths = (),
    # requests_per_second = 2,
    # default_parser = "html.parser",
    # requests_kwargs = None,
    # raise_for_status = False,
    # bs_get_text_kwargs = None,
    # bs_kwargs = None,
    # session = None,
    # show_progress = True,
)

# Load documents
docs = []
docs_lazy = loader.lazy_load()
for doc in docs_lazy:
    docs.append(doc)

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
doc_splits = text_splitter.split_documents(docs)

# Sentence Transformers (faster, good for semantic search)
embeddings = OllamaEmbeddings(
    model="llama3.2"
)

 # Create embeddings for documents and store them in a vector store
vectorstore = SKLearnVectorStore.from_documents(
    documents=doc_splits,
    embedding=embeddings
)
retriever = vectorstore.as_retriever(k = 1)

 # Define the prompt template for the LLM
prompt = PromptTemplate(
    template="""You are an assistant for question-answering tasks.
    Use the following document to answer the question.
    If you don't know the answer, just say that you don't know.
    Use 10 sentences maximum and keep the answer concise:
    Question: {question}
    Documents: {document}
    Answer:
    """,
    input_variables=["question", "document"],
)

# Initialize the LLM with Llama 3.2 model
llm = ChatOllama(
    model="llama3.2",
    temperature=0,
)

# Create a chain combining the prompt template and LLM
rag_chain = prompt | llm | StrOutputParser()

# Define the RAG application class
class RAGApplication:
    ## Define the constructor for the RAG application
    def __init__(self, retriever, rag_chain):
        self.retriever = retriever
        self.rag_chain = rag_chain
    ## Define the run method for the RAG application
    def run(self, question):
        # Retrieve relevant documents
        document = self.retriever.invoke(question)
        # Extract content from retrieved documents
        print("Extracting content from documents...")
        # Get the answer from the language model
        answer = self.rag_chain.invoke({"question": question, "document": document[0].page_content})
        return answer

# Initialize the RAG application
rag_application = RAGApplication(retriever, rag_chain)
