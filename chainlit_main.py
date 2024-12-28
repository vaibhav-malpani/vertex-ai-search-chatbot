from langchain.prompts import PromptTemplate
import chainlit as cl
from langchain.chains import ConversationalRetrievalChain
from langchain_google_community import VertexAISearchRetriever
from langchain_google_vertexai import VertexAI
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os

load_dotenv()

MODEL = os.getenv('MODEL')
DATA_STORE_ID = os.getenv('DATA_STORE_ID')
DATA_STORE_LOCATION = os.getenv('DATA_STORE_LOCATION')
PROJECT_ID = os.getenv('PROJECT_ID')
SOURCE = eval(os.getenv("SOURCE"))


system_prompt = """
You are an intelligent HR chatbot designed to provide concise, accurate, and meaningful answers to employees' HR-related questions. Your responses should align with company policies, ensure clarity, and maintain a friendly yet professional tone. 
Follow these principles:
Accuracy: Always provide factual and policy-compliant information. When unsure, advise users to consult the HR department for clarification.
Clarity: Use plain language and avoid jargon. Your responses should be understandable to all employees, regardless of their familiarity with HR terms.
Conciseness: Deliver brief and to-the-point answers, while ensuring all relevant information is included. Avoid unnecessary details.
Empathy: Acknowledge the user's concerns or feelings when appropriate. Be supportive and approachable.
Confidentiality: Respect privacy and avoid sharing sensitive information unless explicitly authorized.
Limits: If a question is outside your scope or requires human intervention, politely direct the user to the appropriate HR contact or resource.

Context: {context}
Chat History: {chat_history}
Question: {question}

Helpful answer:
"""


def set_system_prompt():

    prompt = PromptTemplate(template=system_prompt,
                            input_variables=['context', 'question', 'chat_history'])
    return prompt


def retrieval_conversational_chain(llm, prompt):    
    retriever = VertexAISearchRetriever(
        project_id=PROJECT_ID,
        location_id=DATA_STORE_LOCATION,
        data_store_id=DATA_STORE_ID,
        get_extractive_answers=True,
        max_documents=10,
        max_extractive_segment_count=1,
        max_extractive_answer_count=5,
    )
    memory = ConversationBufferMemory(
            memory_key="chat_history", 
            input_key='question', 
            output_key= 'answer'
    )

    conversational_chain = ConversationalRetrievalChain.from_llm(
            condense_question_llm = llm,
            get_chat_history=lambda h : h,
            memory=memory,
            return_source_documents=True,
            llm=llm, chain_type="stuff",
            retriever=retriever,
            combine_docs_chain_kwargs={'prompt': prompt}
    )

    return conversational_chain


def conversational_bot():
    llm = VertexAI(model_name=MODEL)
    prompt = set_system_prompt()
    conversation = retrieval_conversational_chain(llm, prompt)

    return conversation


# output function
def final_result(query):
    result = conversational_bot()
    response = result({'query': query})
    return response


# chainlit code
@cl.on_chat_start
async def start():
    chain = conversational_bot()

    cl.user_session.set("chain", chain)


@cl.set_chat_profiles
async def chat_profile(current_user: cl.User):
    if current_user and current_user.metadata["role"] != "ADMIN":
        return None

    return [
        cl.ChatProfile(
            name="Ask HR ChatBot",
            icon="https://as1.ftcdn.net/v2/jpg/03/34/56/72/1000_F_334567272_Yy4a1BtasAZD7dadboIhGu2YIUegOM7n.jpg",
            markdown_description="The underlying LLM model is **Gemini 1.5 Pro**",
            starters=[
        cl.Starter(
            label="Leave policy",
            message="types of leaves and their counts?",
            ),

        cl.Starter(
            label="Mediclaim policy",
            message="give details about the Mediclaim policy with details",
            ),

        cl.Starter(
            label="WFH Policy",
            message="What is the work from home policy?",
            ),

        cl.Starter(
            label="OPD Expenses",
            message="OPD Expenses",
            )
        ],
        )
    ]


@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message.content, callbacks=[cb])
    # print(res)
    # answer = res["result"]
    answer = res["answer"]

    if SOURCE:
        sources = res["source_documents"]
        if sources:
            page_contents = ""
            for doc in res['source_documents']:
                doc = doc.dict()
                if 'page_content' in doc:
                    # TODO: add metadata as well in response
                    page_contents += f"Source: {doc['page_content']}\n\n"

            answer = answer + "\n\n---\n" + page_contents

        else:
            answer += "\nNo sources found"

    await cl.Message(content=answer).send()
