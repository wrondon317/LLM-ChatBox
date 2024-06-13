# LLM-ChatBox
Documentation of Modifications and Implementation
Purpose of the RAG Component
Retrieval-augmented generation (RAG) is a technique that enhances the capabilities of language models by integrating retrieval mechanisms with generative models. In this implementation, RAG is used to improve the accuracy and relevance of responses by retrieving relevant document chunks and using them to generate more informed answers. This is particularly useful for answering questions based on specific documents uploaded by the user.

Documents Used for Augmentation
The documents used for augmentation are those uploaded by the user through the Streamlit sidebar. These documents can be in PDF, TXT, or DOCX format. Once uploaded, they are stored in a specified directory (DATA_PATH), loaded, and processed to create embeddings. These embeddings are then used to retrieve relevant chunks of text when a user asks a question.

Streamlit: Used to build the web application interface.
os: Interacts with the operating system, mainly for file handling.
dotenv: Loads environment variables from a .env file, which is crucial for securely managing API keys.
langchain components: Various modules from langchain are used to handle document loading, splitting, embedding creation, and retrieval.

Loading Environment Variables
Loads environment variables, particularly the OpenAI API key needed for accessing OpenAI's models.

Path and API Key Setup
Defines the path for storing uploaded data files.
Fetches the OpenAI API key from the environment variables and raises an error if it's not found.

Streamlit App Title
Sets the title of the Streamlit app, giving users a clear idea of the application's purpose.

File Upload Handling
Creates the data directory if it doesn't exist.
Handles file uploads via the Streamlit sidebar and saves the uploaded files to the specified directory.

Document Processing and Retrieval Setup
•	Checks if the data directory is empty and prompts the user to upload files if it is.
•	Initializes the language model using the OpenAI API key.
•	Loads and splits documents into manageable chunks.
•	Creates embeddings for the document chunks and stores them in a Chroma vector store.
•	Sets up the vector store as a retriever for fetching relevant document chunks.
•	Retrieves a predefined RAG prompt from the hub.

RAG Chain Definition
•	Defines a function to format document chunks for display.
•	Defines a chain of operations for RAG: retrieving context, formatting documents, applying the prompt, and generating a response.
Session State and Chat Handling
•	Initializes session state for storing chat messages.
•	Displays previous messages in the chat interface.
•	Handles user input through the chat interface and generates responses using the RAG chain.

Challenges Encountered
•	Handling Various Document Formats: Ensuring compatibility with different document types (PDF, TXT, DOCX) and handling their specific parsing requirements.
•	Efficient Document Splitting: Balancing between chunk size and overlap to optimize the retrieval and generation processes without losing context.
•	Integration of Multiple Components: Seamlessly integrating various components from the langchain library to work together in a cohesive manner. Version library installation was the most challenging part, and understand the documentation in lang chain.
•	Managing State in Streamlit: Ensuring that the chat interface correctly maintains the state across user interactions for a smooth user experience.
