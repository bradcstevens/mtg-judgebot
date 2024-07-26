from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def create_retrieval_chain(retriever, openai_api_key):
    template = """Answer the Magic: The Gathering question based on the following context. If the information is not explicitly stated in the context, use your understanding of the game to provide a reasonable answer, but indicate when you're extrapolating beyond the given information.

    Context:
    {context}

    Question: {question}

    Detailed Answer:"""
    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI(api_key=openai_api_key, temperature=0.7)

    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    mtg_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    return mtg_chain