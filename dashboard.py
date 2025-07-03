import streamlit as st
import os 
from retrieval import Search
from google import genai
from google.genai import types
from retrieval import Search
import time 


client = genai.Client(api_key="AIzaSyCEutNRJD7qGDTgxfiW8xPpvqWoZPpAe5k")

search_object = Search()

st.title("RAG-Bot")

uploaded_file = st.sidebar.file_uploader("Upload a File of Size 10")

try:
    if uploaded_file is not None:
        file_path = os.path.join("temp_files", uploaded_file.name)
        
        #make the directory if does not exist
        os.makedirs("temp_files", exist_ok=True)
        
        #Save File to Disk
        with open(file_path,"wb") as f:
            f.write(uploaded_file.read())
        
        st.sidebar.success(f"File Saved at : {file_path}")
        
        search_object.embed_doc(file_path)


    user_prompt = st.text_input("Enter Prompt")

    if user_prompt:
        user_embedding = search_object.embed_user(user_prompt)
        user_context = search_object.search(user_embedding)

        response = client.models.generate_content_stream(
            model="gemini-2.5-flash",
            contents=(
                f" User Prompt : ``` {user_prompt} ``` \n"
                rf"Context :  ``` {user_context} ```"
            ),
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0) ,# Disables thinking
                system_instruction= "You Will Be Given a User Prompt  and A Context For It, If you think the context is relevant for the user query then respond accordingly, otherwise tell that the doc doesn't have answers to the query"
            ),
        )
except IndexError:
    st.write("Upload a doc First Pls")

def stream_data():
    for chunk in response:
        yield chunk.text
        time.sleep(0.1)

try:
    if st.button("Ask"):
        st.write_stream(stream_data)
except NameError:
    st.write("You Have Asked Nothing")

