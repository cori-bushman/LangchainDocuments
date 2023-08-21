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
max_tokens = 400
llm = OpenAI(temperature=0.1, verbose=True, max_tokens=max_tokens)

loader = Docx2txtLoader('NewMSAPlaybook.docx')
pages = loader.load_and_split()
text_splitter = RecursiveCharacterTextSplitter(["\nSection:", "MSA Original Language"])
playbook_docs = loader.load_and_split(text_splitter)

output_parser = RegexParser(
    regex=r".*?Issue:(.*)\nReason:(.*)",
    output_keys=["answer", "reason"],
)

prompt_template = """
MSA Playbook Context: generally acceptable and unacceptable changes
{context}

MSA Section to Review: 
{msa_section}

Query: As a paralegal representing the Service Provider, not the Buyer, identify language in the given MSA section that is unacceptable for the Service Provider based on the playbook's guidelines.
Understand that this review pertains to a specific segment of the MSA. Therefore, missing clauses from the overall MSA should not be considered as issues in this context.
Answer with issue and reason.

Use this exact format to answer:
Issue: [Unacceptable language found]
Reason: [Reason why the language is unacceptable]

If no issues are found, respond with:
Issue: None
Reason: None
"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "msa_section"],
    output_parser=output_parser,
)

question_prompt_template = """Portion of MSA Playbook (generally acceptable and unacceptable changes):
{context}

MSA Section to Review: 
{msa_section}

Query: As a paralegal representing the Service Provider, identify language in the given MSA section that is unacceptable 
for the Service Provider based on the playbook's guidelines.
If no issues are found, don't make up any.

Use this exact format to answer:
Issue: [Unacceptable language found]
Reason: [Reason why the language is unacceptable]
"""
QUESTION_PROMPT = PromptTemplate(
    template=question_prompt_template,
    input_variables=["context", "msa_section"],
    output_parser=output_parser,
)

combine_prompt_template = """
Given the following extracted parts of a MSA playbook and a MSA draft section, create a final answer that lists up to 5 
unique issues with the MSA section, and why the language is unacceptable.
List in order of issue severity, and be concise. If any issues are similar, combine them into a single issue.
Don't make up more issues if less than 5 are found.

MSA Section to Review: 
{msa_section}
=========
{summaries}
=========
Final answer:"""

COMBINE_PROMPT = PromptTemplate(
    template=combine_prompt_template,
    input_variables=["summaries", "msa_section"],
    output_parser=output_parser
)

chain = load_qa_chain(
    llm,
    chain_type="map_reduce", return_map_steps=True, question_prompt=QUESTION_PROMPT, combine_prompt=COMBINE_PROMPT)


st.title('🦜🔗 MSA Review Playbook')
section = st.text_input("MSA Section Text")

if section:
    try:
        answer = chain({"input_documents": playbook_docs, "msa_section": section}, return_only_outputs=True)
        st.write(answer["output_text"])

        with st.expander("See intermediate steps"):
            st.write(answer["intermediate_steps"])

    except ValueError as e:
        st.write("Failed to parse output. Try again.")
        st.write(e)
    except Exception as e:
        st.write(e)
