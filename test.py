import os
from dotenv import load_dotenv
load_dotenv()

print(os.getenv("OPENAI_API_VERSION"))
print(os.getenv("AZURE_OPENAI_ENDPOINT"))
