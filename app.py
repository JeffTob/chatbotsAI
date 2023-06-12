import streamlit as st
import langchain
from langchain.agents import create_csv_agent
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
#from dotenv import load_dotenv
import os
from streamlit_chat import message
import requests
from bs4 import BeautifulSoup
import openai
from pathlib import Path 
#from llama_index import download_loader, GPTSimpleVectorIndex, LLMPredictor, QuestionAnswerPrompt, PromptHelper
from llama_index import download_loader, GPTVectorStoreIndex, LLMPredictor, QuestionAnswerPrompt, PromptHelper

api_key = ""

def main():
    st.sidebar.title("Dashboard")  # Add title to the sidebar

    st.title("Generative AI ChatBots")

    st.sidebar.text("Enter your OpenAI API key to start")  # Add title to the sidebar   

    openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")  # Add OpenAI API Key input field

    if st.sidebar.button("Start"):
        global api_key
        api_key = openai_api_key
        #os.environ["OPENAI_API_KEY"] = openai_api_key  # Set the OpenAI API key in the environment variable

    # Create a sidebar menu
    page = st.sidebar.selectbox("Select a page", ["Chat Support (LangChain)", "Chat Support (LlamaIndex)", "Chat PDF", "Chat MLS"])

    
   

    st.sidebar.title("Bradford Technologies")  # Add title to the sidebar    
    st.sidebar.text("Developed by Jeferson Tobias")  # Add title to the sidebar    

    if page == "Chat Support (LangChain)":
        pageSupport()
    elif page == "Chat Support (LlamaIndex)":
        pageSupport2()    
    elif page == "Chat PDF":
        pagePDF()
    elif page == "Chat MLS":
        pageMLS()


def pageSupport():
    #st.header("Chat Support")
    #st.write("Welcome to Support!")

    #load_dotenv()

    #if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
    if api_key is None or api_key == "":
        st.error("OPENAI_API_KEY is not set")
        return

    st.title("💬 Chat Support (LangChain)")
    st.write("Scraper - BeautifulSoup")

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    with st.form("url_input_form"):
        url_input = st.text_input("Enter URL:", help="Enter the URL of a web page")
        load_button = st.form_submit_button("Load Data")

    if load_button:
        if url_input:
            response = requests.get(url_input)
            soup = BeautifulSoup(response.text, "html.parser")
            page_content = soup.get_text()
            st.session_state["content"] = page_content
            # Clear previous chat messages
            st.session_state["messages"] = []
        

    with st.form("chat_input", clear_on_submit=True):
        a, b = st.columns([4, 1])
        user_input = a.text_input(
            label="Your message:",
            placeholder="What would you like to ask about the URL content?",
            label_visibility="collapsed",
        )
        b.form_submit_button("Send", use_container_width=True)

    if url_input and not api_key:
        st.info("Please add your OpenAI API key to continue.")

    if load_button:
            message('How can I help you?', is_user=False)  # Display AI's response

    if user_input and api_key:
        with st.spinner(text="Thinking..."):
            openai.api_key = api_key
            if "content" in st.session_state:
                st.session_state["messages"].append({"role": "user", "content": user_input})
                filtered_messages = [msg["content"] for msg in st.session_state["messages"] if msg["role"] == "user"]
                filtered_messages.append(st.session_state["content"])
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "assistant", "content": "You are asking about the URL content and content only nothing else."}]
                            + [{"role": "user", "content": msg} for msg in filtered_messages],
                )
                msg = response.choices[0].message
                st.session_state["messages"].append(msg)

                # Check if the response is related to the URL content
                response_content = msg["content"]
                if "I'm sorry, but I don't have information about that in the URL content." in response_content:
                    st.warning("Your question is not related to the URL content.")
                else:
                    message(user_input, is_user=True)
                    message(response_content)
            else:
                st.warning("Please load the URL content first.")

def pageSupport2():
    #st.header("Chat Support LlamaIndex")
    #st.write("Welcome to Support 2!")

    #load_dotenv()
    #openai_api_key = os.getenv("OPENAI_API_KEY")

    if api_key is None or api_key == "":
        st.error("OPENAI_API_KEY is not set")
        return

    st.title("💬 Chat Support (LlamaIndex)")
    st.write("Scraper - BeautifulSoup")

    def main(query, url):
    
      BeautifulSoupWebReader = download_loader("BeautifulSoupWebReader")
      loader = BeautifulSoupWebReader()
      documents = loader.load_data(urls=[url])
        
      QA_PROMPT_TMPL = (
            "We have provided context information below.\n"
            "---------------------\n"
            "{context_str}"
            "\n---------------------\n"
            #"Given this information, please answer the question in bullet point format with a new line after each point and cross reference any data cited in the document.\n"
            "warn the user if any information seems off: {query_str}\n"
        )
      QA_PROMPT = QuestionAnswerPrompt(QA_PROMPT_TMPL)

      # set maximum input size
      max_input_size = 4096
      # set number of output tokens
      num_outputs = 2000
      # set maximum chunk overlap
      max_chunk_overlap = 20
      # set chunk size limit
      chunk_size_limit = 600 

      llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003", openai_api_key=openai_api_key))
      prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)
      #index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)
      index = GPTVectorStoreIndex(documents)
      response = index.query(query, text_qa_template=QA_PROMPT)
      #response = index.query(query,response_mode="tree_summarize")
      return response

    #url_input = st.text_input("Enter a URL")
    with st.form("url_input_form"):
      url_input = st.text_input("Enter URL:", help="Enter the URL of a web page")
      load_button = st.form_submit_button("Load Data")    

    #user_input = st.text_input("Ask a question about the webpage")

    with st.form("chat_input", clear_on_submit=True):
        a, b = st.columns([4, 1])
        user_input = a.text_input(
            label="Your message:",
            placeholder="What would you like to ask about the URL content?",
            label_visibility="collapsed",
        )
        send_button = b.form_submit_button("Send", use_container_width=True)

    if load_button:
        message('How can I help you?', is_user=False)  # Display AI's response

    #if st.button("Send"):
    #  response = main(query=user_input, url=url_input)
    #  message(response, is_user=False)  # Display AI's response

    #if st.button("Send"):
    if send_button:
      with st.spinner(text="Thinking..."):
        response = main(query=user_input, url=url_input)
        response_text = str(response)  # Convert the response to a string
        message(user_input, is_user=True)
        message(response_text, is_user=False)  # Display AI's response

def pagePDF():
    #st.header("Chat PDF")
    #st.write("Welcome to PDF!")

    # Add the code here
    #load_dotenv()

    #if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
    #    st.error("OPENAI_API_KEY is not set")
    #    return

    if api_key is None or api_key == "":
        st.error("OPENAI_API_KEY is not set")
        return
    
    st.title("💬 Chat PDF")
    st.write("LangChain embeddings")

    # upload file
    pdf = st.file_uploader("Upload your PDF", type="pdf")

    # extract the text
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        # create embeddings
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        # show user input
        with st.form("pdf_input_form", clear_on_submit=True):
            a, b = st.columns([4, 1])
            user_question = a.text_input(
                label="Ask a question about your PDF:",
                placeholder="What would you like to ask about the URL content?",
                label_visibility="collapsed",
            )
            b.form_submit_button("Send", use_container_width=True)

        # show user input
        #user_question = st.text_input("Ask a question about your PDF:")
        if user_question:
            docs = knowledge_base.similarity_search(user_question)

            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=user_question)
                with st.spinner(text="Thinking..."):
                    #response_text = str(response)  # Convert the response to a string
                    message(user_question, is_user=True)
                    message(response, is_user=False)  # Display AI's response

                #print(cb)

            #st.write(response)               


def pageMLS():
    #st.header("Chat MLS")
    #st.write("Welcome to MLS!")

    # Add the code here
    #load_dotenv()

    #if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
    #    st.error("OPENAI_API_KEY is not set")
    #    return

    if api_key is None or api_key == "":
        st.error("OPENAI_API_KEY is not set")
        return
    
    st.title("💬 Chat MLS ")
    st.write("LangChain Agent")

    csv_file = st.file_uploader("Upload a CSV file", type="csv")
    if csv_file is not None:
        agent = create_csv_agent(OpenAI(temperature=0), csv_file, verbose=True)

        user_question = st.text_input("Ask a question about your CSV: ")

        if user_question is not None and user_question != "":
            with st.spinner(text="Thinking..."):
                prompt = user_question[:4000]  # Truncate the prompt to 4000 characters
                completion_length = 700  # Set the desired completion length
                completion = agent.run(prompt)[:completion_length]  # Truncate the completion to the desired length
                with st.spinner(text="In progress..."):                    
                    message(prompt, is_user=True)
                    message(completion, is_user=False)  # Display AI's response
                #st.write(completion)


if __name__ == "__main__":
    main()
