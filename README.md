---
title: MedRag
emoji: 🏥
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.52.1
app_file: app.py
pinned: false
---

# MedRag — Multi-Agent Medical Research Assistant

Upload medical research PDFs and ask questions about them. MedRag retrieves the most
relevant passages and answers with a two-agent pipeline: one agent drafts the answer,
a second agent reviews it for accuracy and safety.

**Live demo:** https://huggingface.co/spaces/YOUR-USERNAME/MedRag

## How it works

1. **Classifier agent** checks each PDF and rejects non-medical documents.
2. **Chunk + embed:** text is split into ~400-word chunks and embedded with `all-MiniLM-L6-v2`.
3. **Retrieve:** FAISS finds the 5 chunks closest to the question.
4. **Agent 1** drafts an answer from those chunks.
5. **Agent 2** validates the draft for accuracy, clarity and safety warnings.

Each visitor gets a private, in-memory index, so uploads are never shared between users
and disappear when the session ends. You can open "How this answer was made" under any
answer to see the retrieved passages and the pre-validation draft.

**Stack:** Streamlit · PyMuPDF · sentence-transformers · FAISS · any OpenAI-compatible LLM
(free Groq tier by default)

## Run locally

```bash
git clone https://github.com/YOUR-USERNAME/MedRag.git
cd MedRag
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export LLM_API_KEY="your-groq-key"
streamlit run app.py
```

Sample paper: Son C, et al. *Effects of COVID-19 on College Students' Mental Health in the
United States: Interview Survey Study.* J Med Internet Res 2020 (CC BY 4.0).

*For research and education only. Not medical advice.*
