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

llm = OpenAI(temperature=1, verbose=True)

loader = Docx2txtLoader('MSAReviewPlaybook.docx')
pages = loader.load_and_split()
text_splitter = RecursiveCharacterTextSplitter(["\n\nSection "])
playbook_docs = loader.load_and_split(text_splitter)

output_parser = RegexParser(
    regex=r".*?Issue:(.*)\nReason:(.*)\nScore:(.*)",
    output_keys=["answer", "reason", "score"],
)
# Query: As a paralegal for the Service Provider, what issues can you find in the given MSA section according to the playbook?

prompt_template = """MSA Playbook Context: generally acceptable and unacceptable changes
{context}

MSA Section to Review: 
{msa_section}

Query: As a paralegal representing the Service Provider, identify language in the given MSA section that could pose risks or is unacceptable for the Service Provider based on the playbook's guidelines.
Answer with issue, reason, and score. Use this exact format to answer:
Issue: [Unacceptable Language Issue Found]
Reason: [Reason why the language is unacceptable]
Score: [Score indicating the severity of the issue on a scale of 1 to 100]"""
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

        sorted_steps = sorted(answer["intermediate_steps"], key=lambda x: x["score"], reverse=True)
        for step in answer["intermediate_steps"]:
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
