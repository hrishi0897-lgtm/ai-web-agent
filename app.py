import streamlit as st
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from google import genai
from google.genai import types

st.set_page_config(page_title="AI Web Agent", page_icon="🌐", layout="wide")

st.title("🌐 Autonomous AI Internet Agent")
st.caption("Powered by Gemini with real-time web search and full-page reading capabilities.")

# Sidebar Settings
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Gemini API Key", type="password")
    model_choice = st.selectbox("Select Model", ["gemini-2.5-flash", "gemini-1.5-pro"])
    st.markdown("[Get a free Gemini API Key](https://aistudio.google.com/)")

# Tool Functions
def search_web(query: str) -> str:
    """Searches DuckDuckGo and returns top results."""
    try:
        results = DDGS().text(query, max_results=5)
        if not results:
            return "No search results found."
        formatted = ""
        for r in results:
            formatted += f"- Title: {r.get('title')}\n  URL: {r.get('href')}\n  Snippet: {r.get('body')}\n\n"
        return formatted
    except Exception as e:
        return f"Search error: {str(e)}"

def read_webpage(url: str) -> str:
    """Fetches and extracts clean text from any URL."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        
        # Remove script and style elements
        for element in soup(["script", "style", "nav", "footer"]):
            element.decompose()
            
        text = " ".join(soup.stripped_strings)
        return text[:5000] if text else "Page is empty or protected."
    except Exception as e:
        return f"Failed to fetch webpage: {str(e)}"

# Agent Execution
user_prompt = st.chat_input("Ask your agent to research anything on the web...")

if user_prompt:
    if not api_key:
        st.error("Please enter your Gemini API Key in the sidebar.")
    else:
        st.chat_message("user").write(user_prompt)
        
        client = genai.Client(api_key=api_key)
        
        tools_def = [
            types.Tool(function_declarations=[
                types.FunctionDeclaration(
                    name="search_web",
                    description="Search the internet using DuckDuckGo for live info, facts, and links.",
                    parameters=types.Schema(
                        type="OBJECT",
                        properties={"query": types.Schema(type="STRING")},
                        required=["query"]
                    )
                ),
                types.FunctionDeclaration(
                    name="read_webpage",
                    description="Read the complete text content of a specific web URL.",
                    parameters=types.Schema(
                        type="OBJECT",
                        properties={"url": types.Schema(type="STRING")},
                        required=["url"]
                    )
                )
            ])
        ]
        
        with st.chat_message("assistant"):
            status = st.status("Agent is working...", expanded=True)
            chat = client.chats.create(
                model=model_choice,
                config=types.GenerateContentConfig(
                    tools=tools_def,
                    system_instruction="You are an autonomous internet research agent. Use the search_web and read_webpage tools iteratively to answer user tasks thoroughly."
                )
            )
            
            response = chat.send_message(user_prompt)
            
            # Autonomous execution loop
            steps = 0
            while response.function_calls and steps < 6:
                steps += 1
                call = response.function_calls[0]
                tool_name = call.name
                args = call.args
                
                status.write(f"🔧 **Executing Tool:** `{tool_name}` with arguments: `{args}`")
                
                if tool_name == "search_web":
                    tool_output = search_web(args.get("query"))
                elif tool_name == "read_webpage":
                    tool_output = read_webpage(args.get("url"))
                else:
                    tool_output = "Unknown tool"
                
                response = chat.send_message(
                    types.Part.from_function_response(
                        name=tool_name,
                        response={"result": tool_output}
                    )
                )
            
            status.update(label="Task Complete!", state="complete", expanded=False)
            st.write(response.text)
