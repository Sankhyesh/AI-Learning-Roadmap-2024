- **Core Concept: Structured Output from LLMs**

  - Large Language Models (**LLMs**) traditionally produce **unstructured output**, which is plain text. This is suitable for human-AILM conversation but challenging for direct integration with other software systems.
  - **Structured Output** refers to compelling LLMs to return information in a predefined, well-defined data format, such as **JSON**. This makes the LLM's output programmatically parsable and usable by other systems like databases or APIs.
  - The primary benefit is enabling **LLMs to communicate effectively with other machines**, not just humans, thus expanding their utility significantly.

- **Importance and Use Cases of Structured Output**

  - **Data Extraction**: Systematically pulling specific information from text and organizing it.
    - _Example_: Extracting details like name, last company, and marks from a resume and storing them in a database. The LLM receives the resume text and outputs a JSON with these fields.
  - **API Building**: Creating robust APIs powered by LLMs.
    - _Example_: An API that takes a product review, and the LLM returns a structured JSON containing the review's **topic**, identified **pros**, **cons**, and overall **sentiment**.
  - **Building Agents**: **Agents** are described as "chatbots on steroids" that can perform actions using **Tools**.
    - _Example_: A math agent asked to "find the square root of 2" needs to extract the operation ("square root") and the number ("2") in a structured way to pass to a calculator **Tool**. Tools generally require structured input.
  - This capability is crucial for **LangChain Agents**, a topic highlighted as significant for future learning.

- **Generating Structured Output in LangChain**

  - LangChain provides mechanisms to get structured output from LLMs. The approach depends on the LLM's inherent capabilities:
    - For LLMs that natively support structured output generation (e.g., many OpenAI models), LangChain offers the `with_structured_output` function.
    - For LLMs that do not, LangChain provides **Output Parsers** (to be detailed in subsequent material).
  - The current focus is on the `with_structured_output` method.

- **The `with_structured_output` Function**

  - This function is chained to an LLM model instance before invoking it.
  - It requires a **schema** that defines the desired output structure.
  - Three primary ways to define this schema are discussed:
    1.  **Typed Dictionaries** (from Python's `typing` module)
    2.  **Pydantic Models**
    3.  **JSON Schema**

- **Schema Definition Method 1: Typed Dictionaries**

  - **Typed Dictionaries** (`TypedDict`) allow defining a dictionary's structure with expected keys and their value types.
  - **Purpose**: Primarily for type hinting and code clarity; helps developers understand the expected data structure.
  - **Limitation**: Does not perform runtime data validation. If the LLM returns a type different from what's specified, `TypedDict` itself won't raise an error.
  - **Implementation**:
    - Define a class inheriting from `TypedDict`, specifying attributes and their types (e.g., `summary: str`, `sentiment: str`).
    - Pass this `TypedDict` class to `model.with_structured_output(YourTypedDictClass)`.
  - **Enhancements in LangChain**:
    - **Annotations**: Use `Annotated` from `typing` to add descriptions to fields. This helps the LLM better understand the desired output for each field (e.g., `summary: Annotated[str, "A brief summary of the review"]`).
    - Can handle complex structures like lists (`List[str]`), optional fields (`Optional[List[str]]`), and restricted choices (`Literal["positive", "negative"]`).
  - **Mechanism**: LangChain internally constructs a detailed system prompt based on the provided `TypedDict` schema (and annotations) to guide the LLM to produce the output in the correct JSON format.
  - **Code Example**:

  ```python
  from langchain_openai import ChatOpenAI
  from dotenv import load_dotenv
  from typing import TypedDict, Annotated, Optional, Literal

  load_dotenv()
  model = ChatOpenAI()

  # Define schema using TypedDict
  class Review(TypedDict):
      key_themes: Annotated[list[str], "Write down all the key themes discussed in the review in a list"]
      summary: Annotated[str, "A brief summary of the review"]
      sentiment: Annotated[Literal["pos", "neg"], "Return sentiment of the review either negative, positive or neutral"]
      pros: Annotated[Optional[list[str]], "Write down all the pros inside a list"]
      cons: Annotated[Optional[list[str]], "Write down all the cons inside a list"]
      name: Annotated[Optional[str], "Write the name of the reviewer"]

  # Create structured model
  structured_model = model.with_structured_output(Review)

  # Example usage
  result = structured_model.invoke("""Your product review text here...""")
  ```

- **Schema Definition Method 2: Pydantic Models**

  - **Pydantic** is a Python library for data validation and settings management using Python type annotations.
  - **Purpose**: Enforces data structure, types, and constraints at runtime. Offers robust data validation.
  - **Key Features**:
    - Models inherit from `pydantic.BaseModel`.
    - Attributes are defined with type hints.
    - **Runtime Validation**: Automatically validates incoming data against the schema and raises errors if it doesn't conform.
    - **Optional Fields**: Defined using `typing.Optional` with a default value (often `None`).
    - **Default Values**: Can be set directly for fields.
    - **Type Coercion**: Can automatically convert data to the correct type if possible (e.g., a string "32" to an integer 32 for an `int` field).
    - **Field Function**: Allows adding more detailed validation, descriptions, default values, and constraints (e.g., numeric ranges `gt=0, lt=10`, string patterns via regex).
    - Models can be easily converted to dictionaries (`.model_dump()`) or JSON strings (`.model_dump_json()`).
  - **Implementation**:
    - Define a class inheriting from `BaseModel`.
    - Pass this Pydantic class to `model.with_structured_output()`. The result of the LLM call will be an instance of this Pydantic model.
  - **Advantage**: More powerful than `TypedDict` due to its validation capabilities.
  - **Code Example**:

  ```python
  from langchain_openai import ChatOpenAI
  from dotenv import load_dotenv
  from typing import Optional, Literal
  from pydantic import BaseModel, Field

  load_dotenv()
  model = ChatOpenAI()

  # Define schema using Pydantic
  class Review(BaseModel):
      key_themes: list[str] = Field(description="Write down all the key themes discussed in the review in a list")
      summary: str = Field(description="A brief summary of the review")
      sentiment: Literal["pos", "neg"] = Field(description="Return sentiment of the review either negative, positive or neutral")
      pros: Optional[list[str]] = Field(default=None, description="Write down all the pros inside a list")
      cons: Optional[list[str]] = Field(default=None, description="Write down all the cons inside a list")
      name: Optional[str] = Field(default=None, description="Write the name of the reviewer")

  # Create structured model
  structured_model = model.with_structured_output(Review)

  # Example usage
  result = structured_model.invoke("""Your product review text here...""")
  # Access fields with validation
  print(result.name)  # Typed access with validation
  print(result.model_dump())  # Convert to dictionary
  print(result.model_dump_json())  # Convert to JSON string
  ```

- **Schema Definition Method 3: JSON Schema**

  - **JSON Schema** is a vocabulary that allows you to annotate and validate JSON documents. It's a language-agnostic standard.
  - **Purpose**: Used when a schema needs to be shared or understood across different programming languages or systems.
  - **Structure**: Defined as a JSON object (in Python, a dictionary) containing keys like:
    - `title`: Name of the schema.
    - `description`: Explanation of the schema.
    - `type`: The data type for the schema (e.g., "object" for a JSON object).
    - `properties`: An object where each key is a field name, and its value is a schema defining that field (including its `type`, `description`, and potentially `enum` for restricted values, or `items` for arrays).
    - `required`: An array of strings listing which properties are mandatory.
  - **Implementation**:
    - The JSON schema is defined as a Python dictionary.
    - This dictionary is passed to `model.with_structured_output()`. The LLM output is a Python dictionary.
  - **Code Example**:

  ```python
  from langchain_openai import ChatOpenAI
  from dotenv import load_dotenv

  load_dotenv()
  model = ChatOpenAI()

  # Define schema using JSON Schema
  json_schema = {
    "title": "Review",
    "type": "object",
    "properties": {
      "key_themes": {
        "type": "array",
        "items": {
          "type": "string"
        },
        "description": "Write down all the key themes discussed in the review in a list"
      },
      "summary": {
        "type": "string",
        "description": "A brief summary of the review"
      },
      "sentiment": {
        "type": "string",
        "enum": ["pos", "neg"],
        "description": "Return sentiment of the review either negative, positive or neutral"
      },
      "pros": {
        "type": ["array", "null"],
        "items": {
          "type": "string"
        },
        "description": "Write down all the pros inside a list"
      },
      "cons": {
        "type": ["array", "null"],
        "items": {
          "type": "string"
        },
        "description": "Write down all the cons inside a list"
      },
      "name": {
        "type": ["string", "null"],
        "description": "Write the name of the reviewer"
      }
    },
    "required": ["key_themes", "summary", "sentiment"]
  }

  # Create structured model
  structured_model = model.with_structured_output(json_schema)

  # Example usage
  result = structured_model.invoke("""Your product review text here...""")
  # Result will be a Python dictionary
  ```

- **Choosing the Right Schema Definition Method**

  - **Typed Dictionaries**: Suitable for Python-only projects where only basic type hints are needed without runtime validation.
  - **Pydantic Models**: The recommended "go-to" for Python projects requiring robust data validation, default values, and type coercion.
  - **JSON Schema**: Best for cross-language compatibility or when a standard, language-agnostic schema definition is paramount.

- **Underlying Mechanisms of `with_structured_output`**

  - The function can utilize different **methods** to instruct the LLM:
    - `method="json_mode"`: For LLMs supporting a specific JSON output mode. The LLM is explicitly asked to generate a JSON string. (Often used with models like Claude, Gemini).
    - `method="function_calling"`: Leverages the LLM's function-calling capabilities. The schema is often translated into a function signature that the LLM "tries" to call. (Default for OpenAI models and often more reliable with them).
  - Not all LLMs support these specialized modes. For instance, some open-source models (like the "TinyLlama" example shown via `ChatHuggingFace`) may not support either, and attempts to use `with_structured_output` will fail. Such cases necessitate the use of **Output Parsers**.

- **Looking Ahead**
  - The material concludes by emphasizing the power of structured output.
  - The next topic will be **Output Parsers**, designed to extract structured information from the textual output of LLMs that don't natively support structured data generation modes.

**Stimulating Learning Prompts:**

1.  How might the choice of schema definition (TypedDict, Pydantic, JSON Schema) impact the development workflow and maintainability of a complex application integrating multiple LLM-powered features?
2.  Beyond the use cases mentioned (data extraction, API building, agents), what other novel applications could be unlocked by reliably obtaining structured output from LLMs?
