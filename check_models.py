import requests
from app.config import settings

def main():
    api_key = settings.gemini_api_key
    if not api_key:
        print("❌ Error: No Gemini API key found in your .env file!")
        return

    print("🔍 Asking Google's servers for your available models...\n")
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    response = requests.get(url)

    if response.status_code != 200:
        print(f"❌ Error fetching models: {response.text}")
        return

    data = response.json()
    print("✅ HERE ARE THE EXACT MODEL NAMES YOU CAN TYPE IN YOUR .env FILE:\n")
    print("-" * 50)
    
    count = 0
    for model in data.get("models", []):
        # We only want models that support text/image generation
        methods = model.get("supportedGenerationMethods", [])
        if "generateContent" in methods:
            # Clean up the name (Google returns "models/gemini-..." but we just need "gemini-...")
            name = model.get("name", "").replace("models/", "")
            desc = model.get("description", "No description")
            print(f"👉 {name}")
            print(f"   ({desc})\n")
            count += 1
            
    print("-" * 50)
    print(f"Total available models: {count}")

if __name__ == "__main__":
    main()
