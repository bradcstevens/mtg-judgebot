from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def create_retrieval_chain(retriever, openai_api_key):
    template = """Answer the Magic: The Gathering question based only on the following context:

    {context}

    Question: {question}

    If the question cannot be answered based solely on the provided context, please respond with "I don't have enough information to answer that question."

    Answer:"""
    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI(api_key=openai_api_key)

    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    mtg_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    return mtg_chain