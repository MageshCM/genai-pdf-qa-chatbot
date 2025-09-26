## Development of a PDF-Based Question-Answering Chatbot Using LangChain

### AIM:
To design and implement a question-answering chatbot capable of processing and extracting information from a provided PDF document using LangChain, and to evaluate its effectiveness by testing its responses to diverse queries derived from the document's content.

### PROBLEM STATEMENT:
Design a chatbot that can answer user queries based on the content of an IoT-related PDF document (iot.pdf). The chatbot should provide concise, context-aware responses by leveraging OpenAI embeddings and a retrieval-based approach.

### DESIGN STEPS:

### STEP 1: Load and Process the PDF

Load the iot.pdf file using PyPDFLoader.

Split the document into manageable chunks if necessary.

Ensure that the PDF is loaded successfully and content is accessible for processing.

### STEP 2: Create Embeddings and Vector Store

Use OpenAIEmbeddings to convert the text chunks into vector representations.

Store these embeddings in a vector database (Chroma) for efficient similarity search.

This enables the retrieval of relevant document sections when a question is asked.

### STEP 3: Build and Query the Retrieval-based Chatbot

Define a prompt template to guide the model in generating concise answers with context.

Use ChatOpenAI (GPT-4) as the language model for answering questions.

Set up a RetrievalQA chain that takes a user query, retrieves relevant document sections, and produces a helpful answer.

Query the chatbot with a sample question and display the result.

### PROGRAM:
```
import os
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv, find_dotenv
import openai

# Load API key
_ = load_dotenv(find_dotenv())
openai.api_key = os.environ["OPENAI_API_KEY"]

# Load PDF
file_path = "iot.pdf"
if os.path.isfile(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    print("PDF loaded successfully.")
    print(pages[0].page_content if pages else "PDF is empty.")

# Create embeddings + vectorstore
persist_directory = 'docs/chroma_iot/'
embedding = OpenAIEmbeddings()
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

# Define LLM
llm = ChatOpenAI(model_name='gpt-4', temperature=0)

# Define prompt template
template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 

{context}
Question: {question}
Helpful Answer:"""

QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)

# Create RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

# Interactive input loop
print("\nIoT PDF Chatbot is ready! Type your question (or 'exit' to quit).")
while True:
    question = input("\nYour question: ")
    if question.lower() in ["exit", "quit"]:
        print("Exiting chatbot. Goodbye!")
        break
    result = qa_chain({"query": question})
    print("\nAnswer:", result["result"])

```
### OUTPUT:
<img width="1306" height="273" alt="image" src="https://github.com/user-attachments/assets/ad864b68-61e6-4307-9e54-7fec0d7270ed" />

### RESULT:
Thus, a question-answering chatbot capable of processing and extracting information from a provided PDF document using LangChain was implemented and evaluated for its effectiveness by testing its responses to diverse queries derived from the document's content successfully.


### RESULT:
