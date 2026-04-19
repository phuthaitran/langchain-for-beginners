# 1. Import required modules
import os
from typing import Literal
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# 2. Load environment variables and create the ChatOpenAI model
load_dotenv()

model = ChatOpenAI(
    model=os.getenv("AI_MODEL"),
    base_url=os.getenv("AI_ENDPOINT"),
    api_key=os.getenv("AI_API_KEY"),
    temperature=0
)

# 3. Define a Pydantic model with all required fields:
#    - name: str with Field(description="...")
#    - price: float
#    - category: Literal["Electronics", "Clothing", "Food", "Books", "Home"]
#    - in_stock: bool
#    - rating: float with Field(ge=1, le=5)
#    - features: list[str]
#    Use Field(description="...") to add descriptions for each field
class Product(BaseModel):
    name: str = Field(description="Item's name")
    price: float
    category: Literal["Electronics", "Clothing", "Food", "Books", "Home"]
    in_stock: bool
    rating: float = Field(ge=1, le=5)
    features: list[str]
    
# 4. Create a structured output model using model.with_structured_output(Product)
structured_model = model.with_structured_output(Product)

# 5. Create a prompt template asking to extract product information
template = ChatPromptTemplate.from_messages([
    ("system", "Extract product information from the text"),
    ("human", "{input}")
])
# 6. Create a chain by piping template | structured_model
chain = template | structured_model

# 7. Test with various product descriptions and handle edge cases
#    Access fields using result.name, result.price, etc.
#    Use result.model_dump_json(indent=2) for formatted JSON output
test_products = [
    "MacBook Pro 16-inch with M3 chip, $2,499. Currently in stock. Users rate it 4.8/5. Features: Liquid Retina display, 18-hour battery, 1TB SSD",
    "Cozy wool sweater, blue color, medium size. $89, available now! Customers love it - 4.5 stars. Hand-washable, made in Ireland",
    "The Great Gatsby by F. Scott Fitzgerald. Classic novel, paperback edition for $12.99. In stock. Rated 4.9 stars. 180 pages, published 1925"
]

for product in test_products:
    print(f"\nInput: {product}\n")
    result = chain.invoke({"input": product})
    print(result.model_dump_json(indent=2))
    print(f"Name: {result.name}")
    print(f"Price: {result.price}")
    print(f"Category: {result.category}")
    print(f"In stock: {result.in_stock}")
    print(f"Rating: {result.rating}")
    print(f"Features: {', '.join(result.features)}")
    print()
    print('*' *80)