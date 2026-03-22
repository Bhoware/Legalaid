from django.http import JsonResponse
from .context_extractor import extract_context
from .rights_engine import get_data
from .ai_explainer import explain

def home(request):
    user_input = request.GET.get("query")

    if not user_input:
        return JsonResponse({"error": "No input provided"})

    context = extract_context(user_input)
    case = context.get("crime_type", "unknown")

    if case == "unknown":
        return JsonResponse({"error": "Case not supported yet"})  # ✅ 8 spaces

    data = get_data(case)
    explanation = explain(data)

    return JsonResponse({
        "structured_data": data,
        "explanation": explanation,
        "context": context
    })  # ✅ properly indented inside function
         