import streamlit as st  # Streamlit library for building web apps
import os  # os module for interacting with the operating system
from dotenv import load_dotenv  # Library for loading environment variables from a .env file
from langchain_community.document_loaders import DirectoryLoader  # Document loader from the langchain_community
from langchain import hub  # Hub module from langchain for accessing pre-defined prompts
from langchain_core.output_parsers import StrOutputParser  # Output parser from langchain_core for parsing responses
from langchain_core.runnables import RunnablePassthrough  # Runnable passthrough for creating chain of operations
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Text splitter for splitting documents into chunks
from langchain_openai import OpenAIEmbeddings, ChatOpenAI  # OpenAI integration for embeddings and chat model
from langchain_chroma import Chroma  # Chroma for creating and storing document embeddings

# Load environment variables from .env file
load_dotenv()  # Loads environment variables from a .env file, useful for storing sensitive information like API keys securely.

# Define the path where data will be stored
DATA_PATH = "data/"  # Specifies the directory where uploaded data files will be stored.

# Fetch the OpenAI API key from the environment variable
openai_api_key = os.getenv('OPENAI_API_KEY')  # Retrieves the OpenAI API key from the environment variables.
if not openai_api_key:
    raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")  # Raises an error if the API key is not found.

# Set the title with emoticons on the Streamlit app
st.title("🤖 ChatGPT with RAG (Retrieval-Augmented Generation) WRBOT 📚")  # Sets the title of the Streamlit app with emojis.

# Create a directory for uploaded files if it doesn't exist
os.makedirs(DATA_PATH, exist_ok=True)  # Creates the data directory if it doesn't already exist.

# Handle file uploads through the Streamlit sidebar
upload_file = st.sidebar.file_uploader("Upload your documents here:", type=["pdf", "txt", "docx"], accept_multiple_files=True)  # Allows users to upload documents via the sidebar.
if upload_file:
    for uploaded_file in upload_file:
        file_path = os.path.join(DATA_PATH, uploaded_file.name)
        # Save each uploaded file to the defined data path
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())  # Writes the uploaded file to the specified path.
    st.sidebar.success(f"Saved files: {[file.name for file in upload_file]}")  # Displays a success message with the names of the saved files.

# Check if the DATA_PATH directory is empty
if not os.listdir(DATA_PATH):
    st.warning("Upload a text file to begin")  # Shows a warning if no files are found in the data directory.
else:
    # Initialize the language model with the OpenAI API key
    llm = ChatOpenAI(api_key=openai_api_key, model_name="gpt-3.5-turbo", temperature=0)  # Initializes the ChatGPT model with specific parameters.

    # Load documents from the specified directory
    loader = DirectoryLoader(DATA_PATH)  # Creates a loader to load documents from the data directory.
    docs = loader.load()  # Loads the documents.

    # Split documents into chunks for processing
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Defines a text splitter to split documents into chunks with some overlap.
    splits = text_splitter.split_documents(docs)  # Splits the loaded documents into chunks.

    # Create and store document embeddings
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)  # Creates embeddings for the documents using OpenAI's embedding model.
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)  # Stores the document embeddings in a Chroma vector store.

    # Setup vectorstore as a retriever to fetch relevant document chunks
    retriever = vectorstore.as_retriever()  # Sets up the vector store as a retriever for fetching relevant chunks of documents.

    # Define the RAG (Retrieval-Augmented Generation) prompt from a pre-defined hub
    prompt = hub.pull("rlm/rag-prompt")  # Retrieves a predefined RAG prompt from the hub.

    # Function to format documents for display
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)  # Defines a function to format the documents for display by concatenating their content.

    # Define the RAG chain of operations: retrieving context, formatting, applying the prompt, and generating a response
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )  # Defines a chain of operations for RAG: retrieving context, formatting, applying the prompt, and generating a response.

    # Initialize the session state for storing chat messages
    if "messages" not in st.session_state:
        st.session_state.messages = []  # Initializes a list to store chat messages in the session state.

    # Display previous messages in the chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])  # Displays each previous message in the chat interface.

    # Handle user input through the chat interface
    if user_prompt := st.chat_input("Ask a question based on your documents"):
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)  # Displays the user's input in the chat interface and stores it in the session state.

        # Generate and display a response using the RAG chain
        with st.chat_message("assistant"):
            response = rag_chain.invoke(user_prompt)  # Invokes the RAG chain with the user's input to generate a response.
            st.markdown(response)  # Displays the response in the chat interface.
            st.session_state.messages.append({"role": "assistant", "content": response})  # Stores the response in the session state.
