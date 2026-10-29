from medrag.agents import agent1_primary, agent2_validate


def run_rag_pipeline(store, question):
    context = store.search(question) or "No relevant text found in uploaded documents."
    draft = agent1_primary(context, question)
    final = agent2_validate(draft, question)
    return {"context_used": context, "agent1_reasoning": draft, "final": final}
