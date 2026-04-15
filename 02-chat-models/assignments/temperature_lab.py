# 1. Import required modules
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# 2. Load environment variables
load_dotenv()

# 3. Define a list of temperature values to test [0, 0.5, 1, 1.5, 2]
temperatures = [0, 0.5, 1, 1.5, 2]

# 4. Define your creative prompt
prompt = "Write a tagline for a coffee shop that is no longer than 6 words"

# 5. Loop through each temperature value:
#    - Create a NEW model instance with that temperature
#    - Run 3 trials with the same prompt
#    - Display the results for each trial
for temp in temperatures:
    print(f"\nTemperature: {temp}")
    print("-" *80)
    
    model = ChatOpenAI(
        model=os.getenv("AI_MODEL"),
        base_url=os.getenv("AI_ENDPOINT"),
        api_key=os.getenv("AI_API_KEY"),
        temperature=temp,
    )
    
    try:
        for i in range(1, 4):
            response = model.invoke(prompt)
            print(f"  Try {i}: {response.content}")
    except Exception as error:
        error_msg = str(error)
        if "temperature" in error_msg.lower():
            print(f"  ⚠️  This model doesn't support temperature={temp}. Skipping...")
            print(f"  💡 Error: {error_msg}")
        else:
            # Re-raise unexpected errors
            raise
# 6. Add your analysis comparing the different temperature results