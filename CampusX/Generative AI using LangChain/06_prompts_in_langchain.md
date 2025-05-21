# Prompts in Langchain

- The information presented builds upon previous discussions about Langchain, its components, and specifically the "Models" component, now focusing on **Prompts**.
- A clarification regarding the **LLM `temperature` parameter** was offered:
  - A value near **0** results in **deterministic output**, meaning the same input will consistently produce the same output. This is useful for applications requiring predictability.
  - A value near **2** leads to more **creative or varied output** for the same input, suitable for applications where novelty is desired.

```python
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model='gpt-4', temperature=1.5)

result = model.invoke("Write a 5 line poem on cricket")

print(result.content)
```

- **Prompts** are defined as the instructions or messages sent to a Large Language Model (LLM).
  - While prompts can be **multimodal** (e.g., images, sound), the focus here is on **text-based prompts**, which are currently the most common.
  - The significance of prompts lies in their strong influence on LLM output; even slight changes in a prompt can drastically alter the response. This has given rise to the specialized field of **Prompt Engineering**.
- A key distinction is made between **Static Prompts** and **Dynamic Prompts**:
  - **Static Prompts** are fixed, pre-written prompts.
    - _Contextual Significance:_ While simple, they offer less flexibility and control. If users directly input static prompts, it can lead to inconsistent application behavior, errors from poorly formed queries (e.g., incorrect names in a research summarizer), and a higher chance of the LLM generating undesirable or "hallucinated" content. It becomes difficult to ensure a consistent quality of service or specific output features (like always providing analogies).
  - **Dynamic Prompts** utilize templates with placeholders that are filled with specific values at runtime.
    - _Contextual Significance:_ This approach grants developers more control over the prompt structure, ensuring consistency, better-formed inputs to the LLM, and a more reliable user experience. For instance, a research assistant application could use dropdowns for paper selection, explanation style, and desired length, which then populate a carefully crafted prompt template.
- Langchain provides the **`PromptTemplate` class** for creating dynamic prompts. ![alt text](images/image-8.pngimage-8.png)

  ```python
    from langchain_core.prompts import PromptTemplate
    from langchain_openai import ChatOpenAI
    from dotenv import load_dotenv

    load_dotenv()

    model = ChatOpenAI()

    # detailed way
    template2 = PromptTemplate(
        template='Greet this person in 5 languages. The name of the person is {name}',
        input_variables=['name']
    )

    # fill the values of the placeholders
    prompt = template2.invoke({'name':'sankhyesh'})

    result = model.invoke(prompt)

    print(result.content)
  ```

_ It involves defining a `template` string containing placeholders (e.g., `{variable_name}`) and specifying the `input_variables` that correspond to these placeholders.
_ The template is populated using its `.invoke()` method, passing a dictionary of the variables and their values. \* _Advantages of `PromptTemplate` over simple f-strings_: 1. **Validation**: The `validate_template=True` parameter helps ensure that all defined `input_variables` are present in the template string and vice-versa, catching errors early. 2. **Reusability**: Templates can be saved (e.g., as JSON files using `template.save()`) and loaded elsewhere (`load_prompt()`). This promotes cleaner code and allows for easy sharing and reuse of prompt structures across different parts of an application or even different projects. 3. **Ecosystem Integration**: `PromptTemplate` objects are designed to work seamlessly with other Langchain components, particularly **Chains**. They can be easily piped to models (e.g., `chain = prompt_template | model`), simplifying the workflow of generating a prompt and then passing it to an LLM.

- The construction of a simple **Chatbot** highlights prompt-related challenges and solutions:
  - An initial problem encountered is the chatbot's **lack of memory** or **context**; each interaction is treated as independent.
  - A first attempt to solve this involves maintaining a `chat_history` list, appending both user and AI messages, and sending the entire history with each new query.
  - However, this simple list doesn't distinguish _who_ sent which message (user or AI), which can confuse the LLM as the conversation grows.

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

chat_history = [
    SystemMessage(content='You are a helpful AI assistant')
]

while True:
    user_input = input('You: ')
    chat_history.append(HumanMessage(content=user_input))
    if user_input == 'exit':
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print("AI: ",result.content)

print(chat_history)
```

- To address the ambiguity in chat history, Langchain offers specific **Message Types**:
  - **`SystemMessage`**: Used to set the overall behavior, role, or context for the LLM at the beginning of a conversation (e.g., "You are a helpful assistant specializing in science."). Its significance is in guiding the LLM's persona and response style throughout the interaction.
  - **`HumanMessage`**: Represents messages originating from the human user.
  - **`AIMessage`**: Represents messages generated by the AI.
  - _Contextual Significance:_ Using these distinct message types within the `chat_history` allows the LLM to understand the conversational flow and the roles of participants, leading to more coherent and contextually appropriate responses.

```python
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

messages=[
    SystemMessage(content='You are a helpful assistant'),
    HumanMessage(content='Tell me about LangChain')
]

result = model.invoke(messages)

messages.append(AIMessage(content=result.content))

print(messages)
```

- The material recaps that an LLM's `invoke` function can be used with:
  - A **single message string** for standalone, single-turn queries (which can be static or dynamic via `PromptTemplate`).
  - A **list of structured messages** (using `SystemMessage`, `HumanMessage`, `AIMessage`) for multi-turn conversations like those in chatbots.
- For creating dynamic content within lists of messages (as in multi-turn chat scenarios), Langchain provides the **`ChatPromptTemplate` class**.
  - This class is analogous to `PromptTemplate` but is specifically designed for chat interactions. It allows placeholders within the content of `SystemMessage`, `HumanMessage`, etc.
  - _Contextual Significance:_ It enables the construction of flexible conversational templates where parts of the system instructions or user/AI messages can be dynamically altered based on runtime inputs (e.g., changing the AI's expertise domain or the topic of inquiry dynamically).
  - A specific syntax, often involving passing tuples like `("system", "You are a helpful {domain} expert")` to methods like `ChatPromptTemplate.from_messages([...])`, is highlighted as a reliable way to define these dynamic chat messages.

```python
from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful {domain} expert'),
    ('human', 'Explain in simple terms, what is {topic}')
])

prompt = chat_template.invoke({'domain':'cricket','topic':'Dusra'})

print(prompt)
```

- The **`MessagesPlaceholder` class** is introduced as a special tool for use within a `ChatPromptTemplate`.
  - It acts as a placeholder where an entire list of messages (typically the chat history) can be dynamically inserted at runtime.
  - _Contextual Significance:_ This is crucial for managing ongoing conversations. For example, in a customer support chatbot, the `MessagesPlaceholder(variable_name="chat_history")` would allow the `ChatPromptTemplate` to be populated with all previous messages between the user and the bot for that session, ensuring the LLM has full context before responding to the latest user query. This makes it easier to load and integrate past interactions into the current prompt.

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# chat template
chat_template = ChatPromptTemplate([
    ('system','You are a helpful customer support agent'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human','{query}')
])

chat_history = []
# load chat history
with open('chat_history.txt') as f:
    chat_history.extend(f.readlines())

print(chat_history)

# create prompt
prompt = chat_template.invoke({'chat_history':chat_history, 'query':'Where is my refund'})

print(prompt)
```

Example chat history content (chat_history.txt):

```text
HumanMessage(content="I want to request a refund for my order #12345.")
AIMessage(content="Your refund request for order #12345 has been initiated. It will be processed in 3-5 business days.")
```

Let me also add an example of a more complex prompt template that demonstrates validation and saving functionality:

```python
from langchain_core.prompts import PromptTemplate

# template
template = PromptTemplate(
    template="""
Please summarize the research paper titled "{paper_input}" with the following specifications:
Explanation Style: {style_input}
Explanation Length: {length_input}
1. Mathematical Details:
   - Include relevant mathematical equations if present in the paper.
   - Explain the mathematical concepts using simple, intuitive code snippets where applicable.
2. Analogies:
   - Use relatable analogies to simplify complex ideas.
If certain information is not available in the paper, respond with: "Insufficient information available" instead of guessing.
Ensure the summary is clear, accurate, and aligned with the provided style and length.
""",
input_variables=['paper_input', 'style_input','length_input'],
validate_template=True
)

template.save('template.json')
```

And here's an example of how this template can be used in a simple UI application:

```python
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate, load_prompt

load_dotenv()
model = ChatOpenAI()

st.header('Research Tool')

paper_input = st.selectbox(
    "Select Research Paper Name",
    ["Attention Is All You Need",
     "BERT: Pre-training of Deep Bidirectional Transformers",
     "GPT-3: Language Models are Few-Shot Learners",
     "Diffusion Models Beat GANs on Image Synthesis"]
)

style_input = st.selectbox(
    "Select Explanation Style",
    ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"]
)

length_input = st.selectbox(
    "Select Explanation Length",
    ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"]
)

template = load_prompt('template.json')

if st.button('Summarize'):
    chain = template | model
    result = chain.invoke({
        'paper_input':paper_input,
        'style_input':style_input,
        'length_input':length_input
    })
    st.write(result.content)
```

Stimulating Learning Prompts:

1.  Considering the benefits of `PromptTemplate` and `ChatPromptTemplate`, what kind of safeguards or best practices might a developer implement when defining the _content_ of these templates to prevent misuse or unexpected LLM behavior, especially when parts of the template are user-generated?
2.  How might the concept of `MessagesPlaceholder` be extended or combined with other Langchain features (like memory modules or retrievers) to handle extremely long chat histories or to inject relevant external knowledge into a conversation dynamically?
