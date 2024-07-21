from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain.prompts import PromptTemplate
from langchain.chains import StuffDocumentsChain
from langchain.chains.llm import LLMChain

import os

from secret import api_key

os.environ["USER_AGENT"] = "myagent"
os.environ["GOOGLE_API_KEY"] = api_key

llm = ChatGoogleGenerativeAI(model="gemini-pro")

loader = WebBaseLoader("https://www.geeksforgeeks.org/nodejs/")
docs = loader.load()

template = 'Write a concise summary of the following:\n"{text}"\nCONCISE SUMMARY:'
prompt = PromptTemplate.from_template(template)
llm_chain = LLMChain(llm=llm, prompt=prompt)
stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")

response = stuff_chain.invoke(docs)
# print("_________________")
print(response["output_text"])
# print("_________________")
file = open("output.txt", "w")
file.write(response["output_text"])