# 1. Import required modules
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
import os

# 2. Load environment variables
load_dotenv()

# 3. Create the ChatOpenAI model
model = ChatOpenAI(
    model=os.getenv("AI_MODEL"),
    base_url=os.getenv("AI_ENDPOINT"),
    api_key=os.getenv("AI_API_KEY")
)

# 4. Initialize conversation history list with a SystemMessage for personality
messages = [
    SystemMessage(content="You are a helpful assistant.")
]

# 5. Create a loop that:
#    - Prompts for user input using input()
#    - Adds HumanMessage to messages list
#    - Invokes model with messages list
#    - Adds AIMessage to messages list
#    - Displays the response

while True:
    try:
        prompt = input("\nYou: ").strip()
    except (EOFError, KeyboardInterrupt):
        print(f"Goodbye!\n")
        break
    
    if prompt.lower() == 'quit':
        print("Goodbye!\n")
        break
    
    if not prompt:
        continue
    
    messages.append(HumanMessage(content=str(prompt)))
    
    response = model.invoke(messages)
    print(f"\nAI: {response.content}\n")
    messages.append(AIMessage(content=str(response.content)))

# 6. Check for "quit" to exit the loop

# 7. Show conversation history length on exit
human_chat_length = sum(isinstance(i, HumanMessage) for i in messages)
ai_chat_length = sum(isinstance(i, AIMessage) for i in messages)
print(f"Total messages in history: {len(messages)} messages, that include 1 system message, {human_chat_length} human messages and {ai_chat_length} AI responses")