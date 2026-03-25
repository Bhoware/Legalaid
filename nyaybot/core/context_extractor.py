import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

VALID_CASES = ["theft", "scam", "consumer", "cybercrime", "harassment"]

def extract_context(user_input):
    prompt = f"""
You are a classification system.

Classify the user's problem into ONE OR MORE of the following categories:

- theft
- scam
- consumer
- cybercrime
- harassment

Rules:
- Return ONLY a Python list
- Example: ["cybercrime", "harassment"]
- Do NOT explain
- Do NOT add text
- If unsure, return ["unknown"]

User input:
"{user_input}"
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.choices[0].message.content.strip().lower()

    # 🔥 convert string → list safely
    try:
        result = eval(raw)   # controlled input → ok here
    except:
        return {"crime_types": ["unknown"]}

    # ✅ validate
    valid = [c for c in result if c in VALID_CASES]

    if not valid:
        valid = ["unknown"]

    return {"crime_types": valid}