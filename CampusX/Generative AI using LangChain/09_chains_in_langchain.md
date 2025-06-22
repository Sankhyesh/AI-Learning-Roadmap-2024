## Recap of Previous LangChain Concepts

Before delving into chains, a brief review of previously covered LangChain components is provided to set the context:

- **Models**: Focused on how to interact with different types of AI models.
- **Prompts**: Explored various ways to send inputs to LLMs, including the use of **PromptTemplate**.
- **Structured Output & Output Parsers**: Addressed the significant aspect of generating structured output from LLMs, introducing **Output Parsers** (like **StringOutputParser** and **PydanticOutputParser**) as a means to achieve this. This is highlighted as an important capability, though not a direct LangChain component itself.

---

## Understanding Chains

### What are Chains and Why are They Needed?

- **Core Idea**: LLM-based applications are typically composed of multiple, smaller, interconnected steps. For instance, a simple application might involve:
  1.  Taking a user prompt.
  2.  Sending it to an LLM.
  3.  Processing the LLM's response and displaying it to the user.
- **The Problem with Manual Execution**: Individually executing these steps can be laborious and complex, especially for larger applications. Manually designing prompts, invoking LLMs, and parsing outputs for each step is inefficient.
- **Chains as a Solution**: **Chains** offer a way to create a **pipeline** by connecting these individual steps.
  - **Automatic Data Flow**: The key advantage is that the output of one step in the chain automatically becomes the input for the next.
  - **Simplified Execution**: Once a chain (pipeline) is built, you only need to provide the initial input to the first step and trigger the chain. The entire sequence of operations then executes automatically, yielding the final output.
  - **Significance**: This significantly simplifies the development of LLM applications by abstracting away the manual management of intermediate steps.

### LangChain Expression Language (LCEL)

- The material introduces the **LangChain Expression Language (LCEL)** as the syntax for creating chains.
- A core element of LCEL is the **pipe operator (`|`)**, which is used to connect different components (like prompts, models, and parsers) together to form a chain.
- **Declarative Syntax**: This approach provides a clear and declarative way to define the structure of the pipeline.
- The concept of **Runnables** is mentioned as the underlying mechanism for how chains work and how LCEL facilitates their creation, with a promise of more detail in a future session.

---

## Types of Chains Demonstrated

The material illustrates how to build and use different types of chains with practical examples:

### 1\. Simple Sequential Chain

- **Concept**: A linear sequence of operations.
- **Example**:
  1.  **Prompt**: A **PromptTemplate** is defined to take a `topic` as input and generate a prompt asking for five interesting facts about that topic.
  2.  **Model**: A **ChatOpenAI** model is initialized.
  3.  **Parser**: A **StringOutputParser** is used to get a clean string output from the model.
  4.  **Chain Construction**: `chain = prompt | model | parser`
  5.  **Invocation**: The chain is invoked with a topic (e.g., "cricket"), and it automatically processes the input through the defined steps to produce the facts.
- **Visualization**: The material mentions the `get_graph().print_ascii()` method to visualize the structure of the chain, showing the flow from prompt input to final parsed output.

```python
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = PromptTemplate(
    template='Generate 5 interesting facts about {topic}',
    input_variables=['topic']
)

model = ChatOpenAI()

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({'topic':'cricket'})

print(result)

chain.get_graph().print_ascii()
```

### 2\. More Complex Sequential Chain (Multiple LLM Calls)

- **Concept**: A sequence involving multiple interactions with an LLM or different LLMs.
- **Example Application**:
  1.  User provides a `topic`.
  2.  **First LLM Call**: Generate a detailed report on the topic.
      - `prompt1` (for detailed report) `| model | parser`
  3.  **Second LLM Call**: Take the generated report as input and create a five-point summary.
      - The output of the first sub-chain (the report) is fed into `prompt2` (for summary) `| model | parser`.
- **Chain Construction**: The entire chain is built by sequentially piping these components: `prompt1 | model | parser | prompt2 | model | parser`.
- **Significance**: Demonstrates how longer, multi-stage processes can be elegantly handled.

```python
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

model = ChatOpenAI()

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({'topic': 'Unemployment in India'})

print(result)

chain.get_graph().print_ascii()
```

### 3\. Parallel Chains

- **Concept**: Executing multiple chains concurrently and then combining their results.
- **Key Component**: **RunnableParallel**. This allows for the definition of multiple independent processing paths that operate on the same initial input.
- **Example Application**:
  1.  User provides a large `text` document (e.g., about Support Vector Machines).
  2.  **Parallel Tasks**:
      - **Notes Generation**: `prompt_notes | model1 (e.g., ChatOpenAI) | parser`
      - **Quiz Generation**: `prompt_quiz | model2 (e.g., ChatAnthropic's Claude) | parser`
  3.  **RunnableParallel Structure**:
      ```python
      parallel_chain = RunnableParallel(
          notes=(prompt_notes | model1 | parser),
          quiz=(prompt_quiz | model2 | parser)
      )
      ```
      The keys ("notes", "quiz") in the dictionary passed to `RunnableParallel` define how the outputs of these parallel branches will be structured.
  4.  **Merging**: The outputs (notes and quiz) from `parallel_chain` are then passed to a final chain that merges them into a single document.
      - `prompt_merge | model1 | parser`
  5.  **Overall Chain**: `final_chain = parallel_chain | merge_chain`

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

load_dotenv()

model1 = ChatOpenAI()
model2 = ChatAnthropic(model_name='claude-3-7-sonnet-20250219')

prompt1 = PromptTemplate(
    template='Generate short and simple notes from the following text \n {text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template='Generate 5 short question answers from the following text \n {text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template='Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}',
    input_variables=['notes', 'quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quiz': prompt2 | model2 | parser
})

merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain

# Example Input Text
text = """
Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.

The advantages of support vector machines are:

Effective in high dimensional spaces.

Still effective in cases where number of dimensions is greater than the number of samples.

Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.

Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

The disadvantages of support vector machines include:

If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.

SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below).
"""

result = chain.invoke({'text':text})
print(result)
chain.get_graph().print_ascii()
```

### 4\. Conditional Chains

- **Concept**: Executing different chains based on a specific condition, similar to an if-else statement.
- **Key Component**: **RunnableBranch**. This component takes a series of (condition, chain) tuples and a default chain. It evaluates conditions sequentially and executes the chain corresponding to the first true condition.
- **Example Application**: Sentiment-based feedback response system.
  1.  User provides `feedback` text.
  2.  **Classification Chain**: Determine the sentiment (positive/negative) of the feedback.
      - `prompt_classify | model | parser_pydantic` (using **PydanticOutputParser** to ensure structured output like `{"sentiment": "positive"}` or `{"sentiment": "negative"}`). This step is crucial for reliable conditional logic.
  3.  **Branching Logic**: Based on the classified sentiment:
      - **If Positive**: `prompt_positive_reply | model | parser_string` (generates a thankful reply).
      - **If Negative**: `prompt_negative_reply | model | parser_string` (generates an empathetic reply).
      - **Default**: A **RunnableLambda** can be used to provide a fallback response if neither condition is met (e.g., "Could not find sentiment"). A **RunnableLambda** converts a Python lambda function into a runnable component that can be used within a chain.
  4.  **RunnableBranch Structure**:
      ```python
      branch_chain = RunnableBranch(
          (lambda x: x['sentiment'] == 'positive', positive_reply_chain),
          (lambda x: x['sentiment'] == 'negative', negative_reply_chain),
          default_fallback_chain # (e.g., RunnableLambda(...))
      )
      ```
  5.  **Overall Chain**: `final_chain = classification_chain | branch_chain`
- **Significance**: Enables the creation of dynamic applications that can adapt their behavior based on intermediate results or inputs. The importance of structured output from the classification step for reliable branching is emphasized.

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

model = ChatOpenAI()
parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description='Give the sentiment of the feedback')

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template='Classify the sentiment of the following feedback text into postive or negative \n {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction':parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)

branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'positive', prompt2 | model | parser),
    (lambda x:x.sentiment == 'negative', prompt3 | model | parser),
    RunnableLambda(lambda x: "could not find sentiment")
)

chain = classifier_chain | branch_chain

# Example usage
result = chain.invoke({'feedback': 'This is a beautiful phone'})
print(result)
chain.get_graph().print_ascii()
```

---

## Key Takeaways & Future Outlook

- **Power of Chains**: Chains are a powerful abstraction in LangChain for building complex LLM applications by simplifying the connection and execution of multiple steps.
- **Versatility**: LangChain supports various chain structures, including **sequential**, **parallel**, and **conditional** chains, allowing for flexible application architectures.
- **LCEL and Runnables**: The **LangChain Expression Language (LCEL)** with its pipe operator (`|`) provides an intuitive way to define chains. The underlying concept of **Runnables** (including `RunnableParallel`, `RunnableBranch`, `RunnableLambda`) is foundational to how these chains are constructed and operate.
- **Importance of Parsers**: **Output Parsers** (like `StringOutputParser` and `PydanticOutputParser`) play a vital role, especially in chains that require specific output formats for subsequent steps (e.g., in conditional chains).
- The material concludes by reiterating the importance of these chain types and promising a more in-depth exploration of **Runnables** and the inner workings of chains in the next session, which will further clarify concepts like LCEL.

---

### Stimulating Learning Prompts:

1.  How might the concept of parallel chains be extended to handle more than two concurrent tasks, and what considerations would be important for managing their combined output?
2.  In the conditional chain example, structured output was key. What are other scenarios where ensuring a specific output format from one part of a chain is critical for the successful execution of subsequent parts?
