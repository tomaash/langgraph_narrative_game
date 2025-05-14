import google.generativeai as genai
from config import GEMINI_API_KEY, GENERATE_MODEL_NAME, EVALUATE_MODEL_NAME

generate_model = None
evaluate_model = None

if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        print(f"Gemini API Key configured successfully.")
        
        generate_model = genai.GenerativeModel(GENERATE_MODEL_NAME)
        evaluate_model = genai.GenerativeModel(EVALUATE_MODEL_NAME)
        print(f"Gemini model '{GENERATE_MODEL_NAME}' initialized for generation tasks.")
        print(f"Gemini model '{EVALUATE_MODEL_NAME}' initialized for evaluation and detailed content tasks.")
    except Exception as e:
        print(f"Error configuring Gemini API or initializing model: {e}")
else:
    print("Error: GEMINI_API_KEY not found in config. Core LLM features will be skipped.") 