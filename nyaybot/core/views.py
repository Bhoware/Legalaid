from django.shortcuts import render
from .classifier import classify
from .context_extractor import extract_context
from .rights_engine import get_data
from .ai_explainer import explain


def home(request):
    result = None

    if request.method == "POST":
        user_input = request.POST.get("text", "").strip()

        if user_input:
            # Step 1: Try ML classifier first
            output = classify(user_input)

            if output["source"] == "ml":
                crime_types = [output["crime_type"]]
            else:
                # Step 2: Fall back to keyword-based multi-crime extraction
                context = extract_context(user_input)
                crime_types = context.get("crime_types", ["unknown"])

            # Step 3: Build results for each detected crime
            if crime_types == ["unknown"]:
                result = {
                    "unknown": True,
                    "message": "Could not identify the crime type. Please provide more details — what happened, where, and whether any threats or force were involved.",
                    "example": 'Example: "Someone stole my phone on a bus" or "I was cheated online and money was taken from my account"'
                }
            else:
                crimes = []
                for crime_type in crime_types:
                    data = get_data(crime_type)
                    if data:
                        explanation = explain(data, user_input, crime_type)
                        crimes.append({
                            "crime_type": crime_type.replace("_", " ").title(),
                            "explanation": explanation,
                        })

                result = {
                    "unknown": False,
                    "crimes": crimes,
                }

    return render(request, "index.html", {"result": result})