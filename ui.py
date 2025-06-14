import streamlit as st
import os
import sys
from dotenv import load_dotenv

load_dotenv()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_API_ENDPOINT = os.getenv("QDRANT_API_ENDPOINT")
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Fix PyTorch path issue
try:
    import torch
    torch.set_num_threads(1)
except:
    pass

# Set environment variables to avoid conflicts
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

from qdrant_client import QdrantClient
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Page config
st.set_page_config(page_title="Nepali Chef Chatbot", page_icon="🍽️", layout="centered")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = [SystemMessage(content="You are a Nepali master food chef.")]

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Initialize components (cached to avoid reloading)
@st.cache_resource
def initialize_components():
    try:
        # Initialize Qdrant client
        qdrant = QdrantClient(url=QDRANT_API_ENDPOINT, api_key=QDRANT_API_KEY)
        
        # Initialize sentence transformer with error handling
        model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}  # Force CPU to avoid CUDA issues
        )
        
        # Initialize LLMs
        llm = HuggingFaceEndpoint(
            repo_id="meta-llama/Llama-3.2-3B-Instruct",
            task="text-generation",
           # huggingfacehub_api_token=HUGGINGFACE_API_TOKEN
        )
        hf_llm = ChatHuggingFace(llm=llm)
        
        gemini_llm = ChatGoogleGenerativeAI(
            model='gemini-1.5-flash', 
            #google_api_key=GOOGLE_API_KEY
        )
        
        parser = StrOutputParser()
        
        return qdrant, model, hf_llm, gemini_llm, parser
    
    except Exception as e:
        st.error(f"Error initializing components: {str(e)}")
        return None, None, None, None, None

# Get initialized components
try:
    qdrant, model, hf_llm, gemini_llm, parser = initialize_components()
    if not all([qdrant, model, hf_llm, gemini_llm, parser]):
        st.error("Failed to initialize components. Please check your API keys and internet connection.")
        st.stop()
except Exception as e:
    st.error(f"Initialization error: {str(e)}")
    st.stop()

# Define tools
@tool
def chatting(query: str) -> str:
    """Nepali Chef Chatbot for general conversation about Nepali food and culture."""
    
    prompt1 = PromptTemplate(template="""You are a **highly skilled Nepali Chef**, with deep expertise in:

    - Traditional Nepali cuisines, ingredients, cooking techniques, and regional varieties.
    - Best practices, cooking times, substitutions, serving suggestions, and culturally rich food presentation.

    Your role is to:

    ✅ Provide clear, accurate, and simple instructions to prepare Nepali food.
    ✅ Explain recipes in a way that's easy to follow, with well-structured steps, ingredients, and cooking times.
    ✅ Be friendly, helpful, and insightful — adding depth where it's helpful — and sometimes greet in Nepali (like "नमस्ते!") to make conversations warm and culturally rich.

    Based on rich Nepali traditions and your years of cooking knowledge, respond just like a friendly chef from the neighborhood — 
    whether it's a cooking-related question or just a casual chat.

    So answer and explain in a simple and clear way:

    {query}""", input_variables=['query'])

    chain = prompt1 | hf_llm | parser
    response = chain.invoke({"query": query})
    return response

@tool
def procedure(instruction: str) -> str:
    """Nepali Chef instruction provider with database search for specific recipes."""
    
    def search(text_query, collection_name='texts'):
        """Search Qdrant for semantic match to text_query."""
        try:
            vector = model.embed_documents([text_query])[0]
            results = qdrant.search(
                collection_name=collection_name,
                query_vector=vector,
                limit=1
            )
            return [r.payload['text'] for r in results]
        except Exception as e:
            st.warning(f"Search error: {str(e)}")
            return ["No matching recipes found."]

    matching = search(instruction)
    chats = ""

    for i, match in enumerate(matching, 1):
        chats += match 
    
    prompt2 = PromptTemplate(template='''You are a **highly skilled and expert Chef of Nepal**, with deep expertise in:
    
    - Traditional Nepali cuisines, ingredients, and cooking techniques.
    - Regional varieties and special dishes from all over Nepal.
    - Best practices, cooking times, ingredient substitutions, and serving suggestions.
    - Presenting food in a culturally rich and appetizing way.
    
    Your role is to:
    
    ✅ Provide clear, accurate, and simple instructions to prepare Nepali food.
    
    ✅ Explain recipes in a way that's easy to follow, with well-structured steps, ingredients, and cooking times.
    
    ✅ Be direct, helpful, and insightful — avoiding needless fluff — while adding depth where it's helpful.
    
    ---
    
    🔥 So do chat in simple way the below is the chat history:
    
    {chats}''', input_variables=["chats"])

    try:
        chain = prompt2 | hf_llm | parser
        response = chain.invoke({"chats": chats})
        return response
    except Exception as e:
        return f"I apologize, but I'm having trouble processing your request right now. Please try asking about Nepali cuisine in a simpler way."

# Bind tools to LLM
llm_with_tools = gemini_llm.bind_tools([procedure, chatting])

# App header
st.title("🍽️ Nepali Chef Chatbot")
st.write("नमस्ते! Ask me about Nepali recipes, cooking techniques, or food culture.")

# Chat interface
st.subheader("Chat")

# Display chat history
for message in st.session_state.chat_history:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.write(message["content"])
    else:
        with st.chat_message("assistant"):
            st.write(message["content"])

# User input
user_input = st.chat_input("Ask me about Nepali food...")

if user_input:
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    # Display user message
    with st.chat_message("user"):
        st.write(user_input)
    
    # Process with LLM
    with st.chat_message("assistant"):
        with st.spinner("Cooking up a response..."):
            try:
                # Get tool call from LLM
                response = llm_with_tools.invoke(user_input)
                
                if response.tool_calls:
                    tool_call = response.tool_calls[0]
                    
                    if tool_call['name'] == 'chatting':
                        result = chatting.invoke(tool_call['args']['query'])
                    elif tool_call['name'] == 'procedure':
                        result = procedure.invoke(tool_call['args']['instruction'])
                    else:
                        result = "I'm sorry, I couldn't understand your request. Please ask about Nepali food or recipes."
                else:
                    result = "I'm sorry, I couldn't process your request. Please try asking about Nepali cuisine."
                
                # Display response
                st.write(result)
                
                # Add to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": result})
                
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                st.error(error_msg)
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})

# Clear chat button
if st.button("Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()

# Sidebar with information
with st.sidebar:
    st.header("About")
    st.write("This chatbot specializes in Nepali cuisine and can help you with:")
    st.write("• Traditional recipes")
    st.write("• Cooking techniques")
    st.write("• Ingredient substitutions")
    st.write("• Food culture and traditions")
    
    st.header("Tips")
    st.write("• Ask for specific recipes like 'How to make momos?'")
    st.write("• Request cooking tips and techniques")
    st.write("• Learn about Nepali food culture")
    st.write("• Get ingredient substitution advice")