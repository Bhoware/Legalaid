import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def explain(data):
    prompt = f"""
You are a legal awareness assistant for Indian citizens.

Explain the situation clearly and professionally.

Context:
Description: {data.get("description")}
Rights: {data.get("rights")}
Police Obligations: {data.get("police_obligations")}
Law: {data.get("law")}

STRICT RULES:
- Use simple English (very light Hindi only if needed)
- Do NOT mix languages randomly
- Do NOT repeat points
- Do NOT add extra legal info
- Keep sentences short and clear
- No dramatic or emotional tone

OUTPUT FORMAT (FOLLOW EXACTLY):

Situation:
[1–2 lines summary]

Your Rights:
- point 1
- point 2
- point 3

Police Responsibilities:
- point 1
- point 2

Legal Reference:
- Section: ...
- Punishment: ...

End with 1 simple helpful line.
"""
    response = client.chat.completions.create(
    model="llama-3.1-8b-instant",   # or llama-3.3-70b-versatile
    messages=[{"role": "user", "content": prompt}]
    )


    return response.choices[0].message.content