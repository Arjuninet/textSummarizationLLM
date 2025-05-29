import validators,streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import UnstructuredURLLoader,YoutubeLoader
 
 
#Streamlit App
st.set_page_config(page_title="Lanchain: Summarization Text from YT/Website", page_icon=":book:")
st.header("Summarize URL")
 
with st.sidebar:
    groq_api_key = st.text_input("Enter Groq API Key", type="password")
 
generic_url = st.text_input("URL",label_visibility="collapsed")
 
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")
 
prompt_template = """
Provide a summary of the following content in 300 words in urdu:
Content:{text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
 
 
if st.button("Summarize"):
    ## Validate all the inputs
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please enter all the required fields.")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL.")
    else:
        try:
            with st.spinner("Loading..."):
                ## Loading the website or YT data
                if "youtube.com" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
                else:
                    loader = UnstructuredURLLoader(urls=[generic_url],
                                                   ssl_verify=False,                                                
                                                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"})  
                docs = loader.load()
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.run(docs)
                st.success(output_summary)
        except Exception as e:
            st.exception(f"An error occurred: {e}")
