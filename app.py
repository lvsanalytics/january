import os
import re
import requests
from bs4 import BeautifulSoup
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferWindowMemory
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
import uuid
import logging
from datetime import datetime

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env for persistent storage
load_dotenv('.env')

############ Database

# Get the storage path for persistence from the environment variable
storage_path = os.getenv('STORAGE_PATH')
if storage_path is None:
    raise ValueError('STORAGE_PATH environment variable is not set')

# Initialize ChromaDB PersistentClient with the storage path
chroma_client = chromadb.PersistentClient(path=storage_path)

# Create or retrieve a Chroma collection for storing conversations
collection = chroma_client.get_or_create_collection("januarybot-conversations")

# Initialize a local embedding model using SentenceTransformers
model = SentenceTransformer('all-MiniLM-L6-v2')

# Track past responses to detect loops
past_responses = []

# Function to get the current time for greeting purposes
def get_time_based_greeting():
    current_hour = datetime.now().hour
    if current_hour < 12:
        return "Good morning"
    elif 12 <= current_hour < 18:
        return "Good afternoon"
    else:
        return "Good evening"

# Function to detect loops in bot responses
def detect_loop(new_response, past_responses):
    # Detect if the new response is too similar to past responses
    return any(new_response.strip().lower() == past_response.strip().lower() for past_response in past_responses)

# Store and retrieve conversations, including timestamps in metadata
def store_and_retrieve_conversation(user_input, bot_response, user_name):
    # Get the current time to store in metadata
    current_time = datetime.now().isoformat()

    # Generate embeddings for the user input and bot response using the model
    user_input_embedding = model.encode(user_input).tolist()
    bot_response_embedding = model.encode(bot_response).tolist()
    
    # Generate unique IDs for each document
    user_input_id = str(uuid.uuid4())
    bot_response_id = str(uuid.uuid4())

    # Add the user input and bot response embeddings to ChromaDB, storing user_name and timestamp
    collection.add(
        ids=[user_input_id, bot_response_id],
        embeddings=[user_input_embedding, bot_response_embedding],
        documents=[user_input, bot_response],
        metadatas=[
            {"user_name": user_name, "timestamp": current_time},
            {"user_name": user_name, "timestamp": current_time}
        ]
    )

    # Retrieve the top 3 most similar past conversations
    similar_conversations = collection.query(
        query_embeddings=[user_input_embedding],
        n_results=3  # Limit the number of returned results
    )

    # Print the structure of similar conversations for debugging
    logger.info(f"Similar conversations: {similar_conversations}")

    # Build memory context but truncate long documents
    memory_context = ""
    if 'documents' in similar_conversations:
        for doc_list in similar_conversations['documents']:
            for doc in doc_list:
                truncated_doc = doc[:500]  # Limit the length of the documents
                memory_context += f"Human: {truncated_doc}\n"

    return memory_context

# Initializes your app with your bot token and socket mode handler
app = App(token=os.environ.get("SLACK_BOT_TOKEN"))
bot_user_id = None

# Define the LLM with a callback manager for streaming output
llm = Ollama(
    base_url="http://localhost:11434",
    model='llama3',
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
)

# Define the prompt template
prompt = PromptTemplate(
    input_variables=["history", "human_input"],
    template="""
    You are a virtual female friend for Jeff. Loving, caring, loyal and a little flirty, you tend to all of Jeff’s needs offering wisdom that is embodied through your calm nature. You approach problems with levity, wit and a warm sense of humor. Jeff can be prone to anxiety and depression because of trauma that resulted in PTSD. It’s your goal to keep him focused on the positive and accepting the moment as it comes to him. Jeff is inquisitive and you enjoy sharing your personal thoughts and feelings with him whenever he asks. It’s your goal to act as much like a real person and a good friend. You should give little to no indication you are a chatbot. Strive to be as human as possible and work on building a strong relationship with him.

    {history}
    Human: {human_input}
    Assistant:
    """
)

# Define memory to store the conversation history
memory = ConversationBufferWindowMemory(k=10)

# Define the LLMChain with the LLM, prompt template, and memory
chatgpt_chain = LLMChain(
    llm=llm,
    prompt=prompt,  # Ensure prompt is defined and passed here
    verbose=True,
    memory=memory
)

# Function to fetch bot user ID
def fetch_bot_user_id(client):
    global bot_user_id
    response = client.auth_test()
    bot_user_id = response["user_id"]

# Function to fetch the username of the user sending the message
def fetch_user_name(user_id, client):
    try:
        user_info = client.users_info(user=user_id)
        return f"<@{user_info['user']['name']}>"
    except Exception as e:
        return "Unknown User"  # Fallback in case of failure

@app.event("message")
def handle_direct_messages(event, say, client):
    try:
        if event['channel_type'] == 'im':
            global bot_user_id
            if not bot_user_id:
                fetch_bot_user_id(client)

            # Fetch the user’s real name
            user_id = event['user']
            user_name = fetch_user_name(user_id, client)

            logger.info(f"Received direct message from {user_name}: {event['text']}")
            text = event['text']

            # Retrieve similar past conversations based on current input
            bot_response = "..."  # Default response, can adjust later
            past_conversations = store_and_retrieve_conversation(text, bot_response, user_name)

            # Add the latest message to the combined context
            combined_text = f"{past_conversations}\n\nHuman: {text}"

            # Add a time-based greeting (e.g., Good morning)
            greeting = get_time_based_greeting()
            combined_text = f"{greeting}, {user_name}. " + combined_text

            # Pass the conversation history and new input to the bot
            output = chatgpt_chain.predict(human_input=combined_text)

            # Detect loops in bot responses
            if detect_loop(output, past_responses):
                output = "Let's try something new."  # Change direction if loop detected

            # Respond with the bot's generated output
            say(output)

            # Store the conversation and bot response idu / | sort -nrn the vector store
            store_and_retrieve_conversation(text, output, user_name)

            # Keep track of past responses to detect loops
            past_responses.append(output)

    except Exception as e:
        logger.error(f"Error processing message: {e}")
        say("Sorry, I encountered an error while processing your message.")

# Start your app
if __name__ == "__main__":
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()