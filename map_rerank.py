import os

import pandas as pd
from langchain import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter

from settings import openai_api_key
from langchain.llms import OpenAI
import streamlit as st
from langchain.document_loaders import Docx2txtLoader
from langchain.output_parsers import RegexParser

os.environ['OPENAI_API_KEY'] = openai_api_key

llm = OpenAI(temperature=0.1, verbose=True)

loader = Docx2txtLoader('NewMSAPlaybook.docx')
pages = loader.load_and_split()
text_splitter = RecursiveCharacterTextSplitter(["Section: ", "MSA Original Language"])
playbook_docs = loader.load_and_split(text_splitter)

output_parser = RegexParser(
    regex=r".*?Issue:(.*)\nReason:(.*)\nScore:(.*)",
    output_keys=["answer", "reason", "score"],
)

prompt_template = """
MSA Playbook: generally acceptable and unacceptable changes
{context}

MSA Section to Review: 
{msa_section}

Query: As a paralegal representing the Service Provider, identify language in the MSA section that is unacceptable based on the playbook guidelines.
Answer with issue, reason, and score.

Use this exact format to answer:
Issue: [Unacceptable language found]
Reason: [Why language is unacceptable]
Score: [Severity of the issue, on a scale of 1 to 100]

If no issues are found, respond with:
Issue: None
Reason: None
Score: 0
"""
PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "msa_section"],
    output_parser=output_parser,
)

chain = load_qa_chain(llm, chain_type="map_rerank", return_intermediate_steps=True, prompt=PROMPT)

st.title('🦜🔗 MSA Review Playbook')
section = st.text_input("MSA Section Text")

if section:
    try:
        answer = chain({"input_documents": playbook_docs, "msa_section": section}, return_only_outputs=True)

        # df = pd.DataFrame(answer["intermediate_steps"])
        # sorted_df = df.sort_values(by='score', ascending=False)
        # st.dataframe(sorted_df, hide_index=True, height=500)

        sorted_steps = sorted(answer["intermediate_steps"], key=lambda x: int(x["score"]), reverse=True)
        for step in sorted_steps:
            if step["score"] != "0":
                st.divider()
                st.write(f'Issue: {step["answer"]}')
                st.write(f'Reason: {step["reason"]}')
                st.write(f'Score: {step["score"]}')

    except ValueError as e:
        st.write("Failed to parse output. Try again.")
        st.write(e)
    except Exception as e:
        st.write(e)
