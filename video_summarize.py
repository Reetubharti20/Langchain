from langchain.document_loaders import YoutubeLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.vectorstores import FAISS   # database store 
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings()
video_url = "https://www.youtube.com/watch?v=XfpMkf4rD6E"

def create_vectordb_from_youtube_url(video_url: str) -> FAISS:
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)
    db = FAISS.from_documents(docs,embeddings)
    # return db
    return docs

def get_response_from_query(db,query,k=4):
    # text-davinci can handle 4097 tokens
    # k = tokens/chunk_size = 4097/1000 ~ 4
    docs = db.similarity_search(query,k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    llm = OpenAI(model="text-davinci-003")
    prompt = PromptTemplate(
        input_variables = ['question',docs],
        template = """You are a helpful youtube assistant that can answer the questions about vidoe
        based on the video's transcript.
        Answer the following question: {question}
        by searching the following video transcripts:{docs}
        Only use the factual information from the transcript to answer the question.
        if you feel that you don't have enough information to answer the question,
        say "I don't know". Your answers should be detailed."""
    )

    chain = LLMChain(llm=llm, prompt = prompt)
    response = chain.run(question=query, docs = docs_page_content)
    response = response.replace("\n","")
    
    

#print(create_vectordb_from_youtube_url(video_url))

import streamlit as st
import textwrap
st.title("YouTube Assistant")

with st.sidebar:
    with st.form(key="my_form"):
        youtube_url = st.sidebar.text_area(
            label="What is the YouTube video URL?",
            max_chars=50
        )
        query = st.sidebar.text_area(
            label = "Ask me about the video.",
            max_chars=50,
            key="query"

        )

        submit_button = st.form_submit_button(label="Submit")

if query and youtube_url:
    db = create_vectordb_from_youtube_url(youtube_url)
    response, docs =get_response_from_query(db,query)
    st.subheader("Answer:" )
    st.text(textwrap.fill(response, width=80))
    



