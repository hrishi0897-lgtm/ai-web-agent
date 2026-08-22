import streamlit as st
from google import genai
from google.genai import types

st.set_page_config(page_title="AI Web Agent", page_icon="🌐", layout="wide")

st.title("🌐 Autonomous AI Internet Agent")
st.caption("Powered by Gemini with real-time web search and live streaming.")

# Sidebar Settings
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Gemini API Key", type="password")
    model_choice = st.selectbox(
        "Select Model", 
        ["gemini-3.7-flash", "gemini-3.6-flash"]
    )
    st.markdown("[Get a free Gemini API Key](https://aistudio.google.com/)")

# Chat Interface
user_prompt = st.chat_input("Ask your agent to search and analyze anything on the web...")

if user_prompt:
    if not api_key:
        st.error("Please enter your Gemini API Key in the sidebar.")
    else:
        st.chat_message("user").write(user_prompt)
        
        with st.chat_message("assistant"):
            response_container = st.empty()
            full_text = ""
            
            try:
                client = genai.Client(api_key=api_key)
                
                # Stream the response live
                stream = client.models.generate_content_stream(
                    model=model_choice,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        tools=[types.Tool(google_search=types.GoogleSearch())],
                        system_instruction="You are an autonomous internet agent. Search the live web for up-to-date facts and provide structured answers with citations."
                    )
                )
                
                for chunk in stream:
                    if chunk.text:
                        full_text += chunk.text
                        response_container.markdown(full_text + "▌")
                
                # Final clean text render
                response_container.markdown(full_text if full_text else "No output generated.")

            except Exception as err:
                response_container.empty()
                st.error(f"Error: {str(err)}")
