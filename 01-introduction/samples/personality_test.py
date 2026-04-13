# 1. Import required modules
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

# 2. Load environment variables
load_dotenv()

# 3. Create the ChatOpenAI model (reuse for all personalities)
model = ChatOpenAI(
    model=os.environ.get("AI_MODEL", "gpt-4o-mini"),
    base_url=os.getenv("AI_ENDPOINT"),
    api_key=os.getenv("AI_API_KEY")
)

# 4. Define a list of personalities with name and system prompt
personalities = [
    "You are a pirate. Answer all questions in pirate speak with 'Arrr!' and nautical terms.",
    "You are a professional business analyst. Give precise, data-driven answers.",
    "You are a friendly teacher explaining concepts to 8-year-old children."
]
# 5. Define the question to test
question = "What is artificial intelligence?"

# 6. Loop through each personality:
#    - Create messages list with SystemMessage and HumanMessage
#    - Invoke the model with the messages
#    - Display the response with personality name

for personality in personalities:
    print(f"\n--- {personality[:50]}... ---\n")

    messages = [
        SystemMessage(content=personality),
        HumanMessage(content=question)
    ]

    response = model.invoke(messages)

    print(response.content)