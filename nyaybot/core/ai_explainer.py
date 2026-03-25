import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def explain(data, user_input, case):
    prompt = f"""You are NyayBot, an Indian legal assistant. A user has described their problem and it involves: {case.replace("_", " ").upper()}.

Generate a clear legal explanation using ONLY the data provided below. Do NOT make up any information.

Return ONLY valid HTML (no markdown, no code blocks, no extra text). Use this exact HTML structure:

<div class="llm-section">
  <div class="llm-block">
    <h4>What This Crime Means</h4>
    <ul><li>...</li></ul>
  </div>
  <div class="llm-block">
    <h4>Your Rights</h4>
    <ul><li>...</li></ul>
  </div>
  <div class="llm-block">
    <h4>What You Should Do</h4>
    <ul><li>...</li></ul>
  </div>
  <div class="llm-block">
    <h4>What Police Must Do</h4>
    <ul><li>...</li></ul>
  </div>
  <div class="llm-block">
    <h4>What Authorities Cannot Do</h4>
    <ul><li>...</li></ul>
  </div>
  <div class="llm-block">
    <h4>FIR Rules</h4>
    <ul><li>...</li></ul>
  </div>
  <div class="llm-block">
    <h4>Important Timelines</h4>
    <ul><li>...</li></ul>
  </div>
  <div class="llm-block llm-law">
    <h4>Legal Sections & Punishment</h4>
    <ul><li>...</li></ul>
  </div>
  <div class="llm-block llm-helpline">
    <h4>Helplines</h4>
    <ul><li>...</li></ul>
  </div>
</div>

Rules:
- Only use information from the Data below
- Each <li> must be one clear point
- No paragraphs, no numbering, no markdown
- Do not add any text outside the <div class="llm-section">...</div>

Data:
{data}

User problem:
{user_input}
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.choices[0].message.content.strip()

    # Safety: strip any accidental markdown code fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("html"):
            raw = raw[4:]
    if raw.endswith("```"):
        raw = raw.rsplit("```", 1)[0]

    return raw.strip()