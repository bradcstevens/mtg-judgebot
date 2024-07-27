from langchain_core.runnables import RunnablePassthrough

def create_mtg_chain(cards_retriever, rules_retriever):
    # Function to format retrieved documents
    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    # Step 1: Retrieve card information
    retrieve_cards = RunnablePassthrough() | cards_retriever | format_docs

    # Combine the steps
    chain = {
        "card_context": retrieve_cards,
        "question": RunnablePassthrough()
    }

    return chain