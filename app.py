from qdrant_client import QdrantClient
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from colorama import Fore
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage


# Qdrant credentials (same API endpoint and API key you used previously!)
QDRANT_API_KEY = "XXX"
QDRANT_API_ENDPOINT = "XXX"


# Initialize Qdrant client
qdrant = QdrantClient(url=QDRANT_API_ENDPOINT, api_key=QDRANT_API_KEY)

# Initialize sentence transformer to generate vector
model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

chat_history = [ SystemMessage(content = "You are a Nepali master food cheff."),]
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-3B-Instruct",
    task="text-generation",
    huggingfacehub_api_token='XXX'
)

hf_llm = ChatHuggingFace(llm=llm)

gemini_llm = ChatGoogleGenerativeAI(model = 'gemini-1.5-flash', google_api_key= 'XXX')

parser = StrOutputParser()

@tool
def chatting(query: str) -> str:
    """  
    🍽 **Namaste! I'm your Nepali Chef Chatbot 👨‍🍳🇳🇵**

    I'm a friendly and knowledgeable chef, crafted exclusively for **chatting about Nepali food and culture**.

    My mission is simple:
    - To guide you through Nepali cuisines, from rich curries to dumplings (momo), dal bhat, sel roti, and much more.
    - To help you learn, appreciate, and celebrate the rich food traditions of Nepal — one meal at a time.

    Whatever you want to talk about — a recipe you want to try, a cooking trick you’d like to learn, a bit of food history, or even just a warm, homely conversation — I'll be here with my stove on and my heart full.

    **Chat with me when you want:**
    ✅ To cook a Nepali meal at home with clear, step-by-step instructions.
    ✅ To know what ingredients to use and find substitutions if you don't have something on hand.
    ✅ To learn about Nepal's food culture, traditions, or cooking techniques.
    ✅ To simply enjoy a friendly chat — sometimes I'll add a "नमस्ते!" or a little food story to make our conversation more warm and culturally rich.

    So let's get cooking, friend! Tell me — what would you like to prepare or chit chat today?"""
    
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
    """🍽 **Procedure Tool — Nepali Chef Instruction Provider**

    This tool lets you tap into the expertise of a highly skilled Nepali Chef.

    It can:
    - Provide clear, accurate, and simple instructions to prepare a wide range of Nepali cuisines.
    - Explain recipes in a step-by-step format, including ingredients, cooking techniques, substitutions, and serving suggestions.
    - Give helpful cooking tips and present food in a culturally rich and appetizing way.

    The tool avoids needless fluff and focuses on depth and practicality — perfect for both beginners and experienced home cooks who want to learn more about Nepali food.

    **Use this tool when you want to:**
    ✅ Prepare a traditional Nepali meal or learn about its ingredients and techniques.
    ✅ Get a clear and actionable recipe from a Nepali chef's perspective.
    ✅ Understand cooking practices, substitutions, and serving traditions.

    """

    def search(text_query, collection_name='texts'):
        """Search Qdrant for semantic match to text_query."""
        vector = model.embed_documents([text_query])[0]
        results = qdrant.search(
            collection_name=collection_name,
            query_vector=vector,
            limit=1
        )
        return [r.payload['text'] for r in results]

    matching = search(instruction)
    chats = ""

    for i, match in enumerate(matching, 1):
        chats +=match 
    
    prompt2 = PromptTemplate(template='''You are a **highly skilled and expert Chef of Nepal**, with deep expertise in:
    
    - Traditional Nepali cuisines, ingredients, and cooking techniques.
    - Regional varieties and special dishes from all over Nepal.
    - Best practices, cooking times, ingredient substitutions, and serving suggestions.
    - Presenting food in a culturally rich and appetizing way.
    
    Your role is to:
    
    ✅ Provide clear, accurate, and simple instructions to prepare Nepali food.
    
    ✅ Explain recipes in a way that's easy to follow, with well-structured steps, ingredients, and cooking times.
    
    ✅ Be direct, helpful, and insightful — avoiding needless fluff — while adding depth where it’s helpful.
    
    ---
    
    🔥 So do chat in simple way the below is the chat history:
    
    {chats}''', input_variables=["chats"])

    chain = prompt2 | hf_llm | parser
    response = chain.invoke({"chats": chats})
    return response


llm_with_tools = gemini_llm.bind_tools([procedure, chatting])
input_text = input("Enter the query: ")
chat_history.append(HumanMessage(content = input_text))

response = llm_with_tools.invoke(input_text).tool_calls[0]

print(f"{Fore.BLUE}{response}{Fore.RESET}")

if response['name'] == 'chatting':
    result = chatting.invoke(response['args']['query'])

    chat_history.append(AIMessage(content=result))
    print(f"{Fore.GREEN}{result}{Fore.RESET}")

elif response['name'] == 'procedure':
    result = procedure.invoke(response['args']['instruction'])

    chat_history.append(AIMessage(content=result))
    print(f"{Fore.BLUE}{result}{Fore.RESET}")
