import requests
from rest_framework.decorators import api_view
from rest_framework.response import Response

# DGX server IP
FASTAPI_URL = "http://202.92.159.242:8001/predict/"

@api_view(['POST'])  # Ensure it only allows POST
def predict_task_duration(request):
    task_name = request.data.get('task_name', '')

    if not task_name:
        return Response({"error": "Task name is required"}, status=400)

    try:
        response = requests.post(FASTAPI_URL, json={"task_name": task_name})
        response.raise_for_status()  # Raise error for non-200 responses
        return Response(response.json())
    except requests.exceptions.RequestException as e:
        return Response({"error": str(e)}, status=500)
