import os
from langchain import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

from settings import openai_api_key
from langchain.llms import OpenAI
import streamlit as st
from langchain.document_loaders import Docx2txtLoader
from langchain.output_parsers import RegexParser

os.environ['OPENAI_API_KEY'] = openai_api_key

llm = OpenAI(temperature=1, verbose=True)

loader = Docx2txtLoader('MSAReviewPlaybook.docx')
pages = loader.load_and_split()
playbook_docs = loader.load_and_split()

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

        # for step in answer["intermediate_steps"]:
        #     if step["score"] != "0":
        #         st.write(f'Issue: {step["answer"]}\nReason: {step["reason"]}')

        st.write(answer)

    except Exception as e:
        st.write(e)

