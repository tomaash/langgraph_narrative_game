import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access your API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Model names
GENERATE_MODEL_NAME = os.getenv("GENERATE_MODEL_NAME", "gemini-1.5-flash-latest") # Model for story generation
EVALUATE_MODEL_NAME = os.getenv("EVALUATE_MODEL_NAME", "gemini-1.5-flash-latest") # Model for user input evaluation 