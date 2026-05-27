import requests
from config import GEMINI_API_KEY


def list_gemini_models():
    # Google's REST endpoint for listing models
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={GEMINI_API_KEY}"

    print("Fetching available models from Google...\n")
    response = requests.get(url)

    if response.status_code == 200:
        models = response.json().get('models', [])

        print("Available Text/Chat Models:")
        print("-" * 40)

        for model in models:
            # We only care about models that support 'generateContent' (which CrewAI needs)
            if 'generateContent' in model.get('supportedGenerationMethods', []):
                # Strip the "models/" prefix so you can see the exact string for your code
                clean_name = model['name'].replace('models/', '')
                print(f"• {clean_name}")

    else:
        print(f"Failed to fetch models. API returned: {response.status_code}")
        print(response.json())


if __name__ == "__main__":
    list_gemini_models()