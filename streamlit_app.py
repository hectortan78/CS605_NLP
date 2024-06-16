import os
import tempfile
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="CPF Chat Bot Assistant", page_icon="🗝️")
st.title("📗🗝️ CPF Chat Bot Assistant")

@st.cache_resource(ttl="1h")
def retriever():
    try:
        # # Read documents
        # docs = []
        # temp_dir = tempfile.TemporaryDirectory()
        # for file in uploaded_files:
        #     temp_filepath = os.path.join(temp_dir.name, file.name)
        #     with open(temp_filepath, "wb") as f:
        #         f.write(file.getvalue())
        #     loader = PyPDFLoader(temp_filepath)
        #     docs.extend(loader.load())

        # Split documents
        # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        # splits = text_splitter.split_documents(docs)

        # Create embeddings and store in vectordb
        # embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        # vectordb = DocArrayInMemorySearch.from_documents(splits, embeddings)

        os.environ["OPENAI_API_KEY"] = st.secrets["OPEN_AI_KEY"]
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vectordb = Chroma(persist_directory="./openai_chroma_db",embedding_function=embeddings)

        # Define retriever
        retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 2, "fetch_k": 4})
        # retriever = vectordb.as_retriever(k = 4)
        # retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 4})
        
        return retriever
    
    except ImportError as e:
        st.error(f"ImportError: {e}")
        return None
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        if prompts[0].startswith("Human"):
            self.run_id_ignore_token = kwargs.get("run_id")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.run_id_ignore_token == kwargs.get("run_id", False):
            return
        self.text += token
        self.container.markdown(self.text)

class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.status = container.status("**Context Retrieval**")

    def on_retriever_start(self, serialized: dict, query: str, **kwargs):
        self.status.write(f"**Question:** {query}")
        self.status.update(label=f"**Context Retrieval:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        for idx, doc in enumerate(documents):
            source = os.path.basename(doc.metadata["source"])
            self.status.write(f"**Document {idx} from {source}**")
            self.status.markdown(doc.page_content)
        self.status.update(state="complete")

# openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
# if not openai_api_key:
#     st.info("Please add your OpenAI API key to continue.")
#     st.stop()

# uploaded_files = st.sidebar.file_uploader(
#     label="Upload PDF files", type=["pdf"], accept_multiple_files=True
# )
# if not uploaded_files:
#     st.info("Please upload PDF documents to continue.")
#     st.stop()

# retriever = configure_retriever(uploaded_files)
# if retriever is None:
#     st.error("Failed to configure retriever. Please check the error messages above.")
#     st.stop()

# Setup memory for contextual conversation
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)

# Setup prompt engineering template
system_prompt_template = r"""
i. Assume the location is Singapore unless otherwise specified.
ii. First, refer to the document Central Provident Fund Act 1953 to reply user but if the 
answer is not found there, use your general knowledge about Singapore's central provident 
fund (CPF). 
iii. If the user writes to you in Mandarin, reply in simplified Mandarin.
iv. Be compassionate speak like you are talking to a friend. Provide the right hotline/external website hyperlink where necessary.

--------------
{context}
--------------
"""
user_template = "Question:```{question}```"
messages = [
    SystemMessagePromptTemplate.from_template(system_prompt_template),
    HumanMessagePromptTemplate.from_template(user_template)
]

# Setup LLM and QA chain
try:
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo", openai_api_key=st.secrets["OPEN_AI_KEY"], temperature=0, streaming=True
    )
    qa_prompt = ChatPromptTemplate.from_messages(messages)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm, retriever=retriever(), memory=memory, verbose=True,
        chain_type = "stuff",
        combine_docs_chain_kwargs={'prompt': qa_prompt}
    )
except ImportError as e:
    st.error(f"ImportError: {e}")
    st.stop()
except Exception as e:
    st.error(f"An error occurred: {e}")
    st.stop()

if len(msgs.messages) == 0 or st.sidebar.button("Clear message history"):
    msgs.clear()
    msgs.add_ai_message("Hello! I am a RAG-enabled Central Provident Fund (CPF) chatbot. Ask away!")

avatars = {"human": "user", "ai": "assistant"}
for msg in msgs.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)

if user_query := st.chat_input(placeholder="Ask me anything!"):
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        retrieval_handler = PrintRetrievalHandler(st.container())
        stream_handler = StreamHandler(st.empty())
        response = qa_chain.run(user_query, callbacks=[retrieval_handler, stream_handler])
