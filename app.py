import os
from io import StringIO

import docx
from langchain.text_splitter import RecursiveCharacterTextSplitter

from settings import openai_api_key
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
import streamlit as st

from langchain.document_loaders import Docx2txtLoader
from langchain.vectorstores import Chroma

from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)
from docx import Document

os.environ['OPENAI_API_KEY'] = openai_api_key

# Create instance of OpenAI LLM
llm = OpenAI(temperature=1, verbose=True)
embeddings = OpenAIEmbeddings()


# Create and load PDF Loader
loader = Docx2txtLoader('MSAReviewPlaybook.docx')
# Split pages from pdf
pages = loader.load_and_split()
# Load documents into vector database aka ChromaDB
store = Chroma.from_documents(pages, embeddings, collection_name='msa_playbook')

# Create vectorstore info object - metadata repo?
vectorstore_info = VectorStoreInfo(
    name="msa_playbook",
    description="contract rules",
    vectorstore=store
)
# Convert the document store into a langchain toolkit
toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

# Add the toolkit to an end-to-end LC
agent_executor = create_vectorstore_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    prefix='The following are sections of an MSA draft. Compare this document with the MSA Review Playbook and list any potential issues with the draft:'
)
st.title('🦜🔗 MSA Review Playbook')
# Create a text input box for the user
uploaded_file = st.file_uploader('Submit MSA Draft (docx)', type=['docx'])

if uploaded_file:
    try:
        bytes_data = uploaded_file.getvalue()

        doc: Document = Document(uploaded_file)

        chunks = []
        print("STARTING")

        # "Here are chunks of a MSA file. Once I input 'END OF FILE', compare with the MSA playbook and return any potential problems"
        agent_executor.run('Here are chunks of a MSA file. Once I input "END OF FILE", compare with the MSA playbook and return any potential problems')

        count = 0
        sub = 0
        for paragraph in doc.paragraphs:
            print(f'PROGRESS: {count}.{sub}')
            text = paragraph.text
            if not text.isspace() and not text == "":
                if len(text) > 100:
                    for s in text.split('.'):
                        sub += 1
                        agent_executor.run(s)
                else:
                    agent_executor.run(text)
            count += 1
            sub = 0

        response = agent_executor.run('END OF FILE')
        st.write(response)
    except Exception as e:
        st.write(e)

