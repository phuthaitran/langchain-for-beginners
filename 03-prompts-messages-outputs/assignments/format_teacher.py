# 1. Import required modules
import json
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_openai import ChatOpenAI

# 2. Load environment variables and create model with temperature 0
load_dotenv()

model = ChatOpenAI(
    model=os.getenv("AI_MODEL"),
    base_url=os.getenv("AI_ENDPOINT"),
    api_key=os.getenv("AI_API_KEY"),
    temperature=0
)

# 3. Define your teaching examples list with input/output pairs
#    - Each example should show a product description as input
#    - And the corresponding JSON format as output (use json.dumps for formatting)
examples = [
    {
        "input": "Premium wireless headphones with noise cancellation, $199",
        "output": json.dumps(
            {
                "name": "Wireless headphones",
                "price": "$199",
                "category": "Electronics",
                "highlight": "Noise Cancellation"
            }
        )
    },
    {
        "input": "Organic cotton t-shirt in blue, comfortable fit, $29.99",
        "output": json.dumps(
            {
                "name": "Organic cotton T-shirt",
                "price": "$29.99",
                "category": "Clothing",
                "highlight": "Organic cotton, Comfortable fit"
            }
        )
    },
    {
        "input": "Gaming laptop with RTX 4070, 32GB RAM, $1,499",
        "output": json.dumps(
            {
                "name": "Gaming laptop",
                "price": "$1,499",
                "category": "Computers",
                "highlight": "RTX 4070, 32GB RAM"
            }
        )
    }
]

# 4. Create an example template using ChatPromptTemplate.from_messages
#    with ("human", "{input}") and ("ai", "{output}")
example_template = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}"),
])

# 5. Create a FewShotChatMessagePromptTemplate with your examples
few_shot_template = FewShotChatMessagePromptTemplate(
    example_prompt=example_template,
    examples=examples
)

# 6. Build a final prompt that includes the few-shot template
final_template = ChatPromptTemplate.from_messages([
    ("system", "Convert product descriptions into JSON output based on the examples"),
    few_shot_template,
    ("human", "{input}")
])
# 7. Test with new product descriptions and parse the JSON output with json.loads()
chain = final_template | model

test_product = "MacBook Pro 16-inch with M3 chip, $2,499. Features: Liquid Retina display, 18-hour battery, 1TB SSD"
result = chain.invoke({"input": test_product})
print(f"Input: {test_product}")
try:
    # Parse to validate JSON
    parsed = json.loads(str(result.content))
    print("✅ Valid JSON output:")
    print(json.dumps(parsed, indent=2))

    # Validate structure
    required_fields = ["name", "price", "category", "highlight"]
    has_all_fields = all(field in parsed for field in required_fields)

    if has_all_fields:
        print("\n✅ All required fields present")
    else:
        print("\n⚠️  Warning: Missing some required fields")
except json.JSONDecodeError:
    print("❌ Invalid JSON output:")
    print(result.content)