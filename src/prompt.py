from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI


# Prompt Template
prompt_template='''
Use the following pieces of information to answer the users questions.
if you dont know the answer , just say that you dont know . dont try to 
make up the answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful Answer :
'''



