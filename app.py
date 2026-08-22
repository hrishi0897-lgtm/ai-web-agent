import streamlit as st
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from google import genai
from google.genai import types

st.set_page_config(page_title="AI Web Agent", page_icon="🌐", layout="wide")

st.title("🌐 Autonomous AI Internet Agent")
st.caption("Powered by Gemini with real-time web search and reading capabilities.")

# Sidebar Settings
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Gemini API Key", type="password")
    model_choice = st.selectbox(
        "Select Model", 
        ["gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.1-pro-preview"]
    )
    st.markdown("[Get a free Gemini API Key](https://aistudio.google.com/)")

# Tool 1: Web Search
def search_web(query: str) -> str:
    """Searches DuckDuckGo and returns the top 5 live results with snippets and links.
    
    Args:
        query: The search query string to look up on the internet.
    """
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

# Tool 2: Web Scraper
def read_webpage(url: str) -> str:
    """Fetches and extracts full text content from any website URL.
    
    Args:
        url: The exact webpage URL to read.
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        
        for element in soup(["script", "style", "nav", "footer"]):
            element.decompose()
            
        text = " ".join(soup.stripped_strings)
        return text[:8000] if text else "Page is empty or protected."
    except Exception as e:
        return f"Failed to fetch webpage: {str(e)}"

# Chat interface
user_prompt = st.chat_input("Ask your agent to research anything...")

if user_prompt:
    if not api_key:
        st.error("Please enter your Gemini API Key in the sidebar.")
    else:
        st.chat_message("user").write(user_prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Agent is researching the live web..."):
                try:
                    client = genai.Client(api_key=api_key)
                    
                    response = client.models.generate_content(
                        model=model_choice,
                        contents=user_prompt,
                        config=types.GenerateContentConfig(
                            tools=[search_web, read_webpage],
                            system_instruction="You are an autonomous internet research agent. Always use the search_web or read_webpage tools to look up real-time information before answering."
                        )
                    )
                    st.write(response.text)
                except Exception as err:
                    st.error(f"Error: {str(err)}")
