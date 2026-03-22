import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

VALID_CASES = ["theft", "scam", "consumer"]

def extract_context(user_input):
    prompt = f"""
    Extract structured information from this legal problem.
    Input: {user_input}
    Return JSON with:
    - crime_type (theft, scam, consumer, unknown)
    - known_person (yes/no/unknown)
    - urgency (low/normal/high)
    ONLY return valid JSON. No explanation.
    """

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",  # ✅ fixed model
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.choices[0].message.content.strip()

    # ✅ safely parse JSON
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        return {
            "crime_type": "unknown",
            "known_person": "unknown",
            "urgency": "normal"
        }

    # ✅ validate crime_type
    if result.get("crime_type") not in VALID_CASES:
        result["crime_type"] = "unknown"

    return result