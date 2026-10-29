from openai import OpenAI
from medrag.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL_FAST, LLM_MODEL_STRONG

_client = None


def client():
    global _client
    if _client is None:
        if not LLM_API_KEY:
            raise RuntimeError("LLM_API_KEY is not set. Add it as a secret in your Space settings.")
        _client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    return _client


def _ask(model, prompt):
    res = client().chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}], temperature=0.2
    )
    return res.choices[0].message.content


def agent1_primary(context, question):
    """Primary reasoning agent: answers from the retrieved PDF context."""
    return _ask(LLM_MODEL_FAST, f"""You are a medical research assistant.
Use the context extracted from the uploaded PDFs to answer the user's question.

Context from documents:
{context}

User question:
{question}

Your task:
- Understand the user's intent and give the most relevant, accurate answer.
- Summarize key findings when helpful.
- If the answer is not in the documents, say so clearly before giving general medical insight.

Produce a clear, structured response.""")


def agent2_validate(answer, question):
    """Safety + accuracy validator that reviews Agent 1's answer."""
    return _ask(LLM_MODEL_STRONG, f"""You are a medical safety and accuracy validation agent.
Review the answer below for correctness, safety, and clarity.

User question:
{question}

Initial answer:
{answer}

Your task:
- Fix any medical inaccuracies.
- Improve clarity.
- Add safety warnings where needed and remind the reader this is not medical advice.
- Keep it aligned with evidence-based medical information.

Return only the corrected final answer.""")


def classification_agent(text: str) -> bool:
    """Returns True if the document is medical/health/biomedical, False otherwise."""
    if text == "IMAGE_ONLY_PDF":
        return False
    out = _ask(LLM_MODEL_FAST, f"""You are a MEDICAL DOCUMENT CLASSIFICATION AGENT.
MEDICAL: medical or biomedical research papers, clinical case reports, diagnostic summaries,
mental health reports, public health research, clinical trial documents, physician reports.
NON-MEDICAL: leases, real estate, business/finance, legal contracts, school assignments,
marketing, or anything clearly unrelated to health, medicine, biology or clinical care.
If the document contains significant medical content, classify it as medical.

Document text:
{text[:5000]}

Respond with exactly ONE word: medical or non-medical""")
    return out.lower().strip().strip(".").startswith("medical")
