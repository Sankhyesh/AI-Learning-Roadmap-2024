from langchain_community.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize the HuggingFace model
llm = HuggingFaceHub(
    repo_id="google/flan-t5-base",  # Using a smaller, more reliable model
    huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
    model_kwargs={"temperature": 0.5, "max_length": 64}
)
    print("\n--- Testing InferenceClient.chat_completion with mistralai/Mistral-7B-Instruct-v0.1 ---")
    try:
        # Explicitly pass the token here
        client_chat = InferenceClient(model="mistralai/Mistral-7B-Instruct-v0.1", token=token)
        
        messages = [{"role": "user", "content": "What is the capital of France?"}]
        
        # The chat_completion method might return a generator or a direct response
        # depending on the stream parameter (default is False)
        response = client_chat.chat_completion(messages=messages, max_tokens=50) # Use max_tokens or max_new_tokens based on client version
        
        # For huggingface_hub >= 0.22, the response is an object
        # For older versions, it might be a dict. Let's try to access 'generated_text' or print the whole thing.
        if hasattr(response, 'choices') and response.choices:
            print("Mistral chat_completion response (via direct client):", response.choices[0].message.content)
        elif hasattr(response, 'generated_text'): # Some older or different response structures
             print("Mistral chat_completion response (via direct client):", response.generated_text)
        else:
            print("Mistral chat_completion response (via direct client):", response) # Print the whole response object

    except StopIteration as si:
        print(f"StopIteration during Mistral chat_completion (direct client): {si}")
        print("This indicates a failure in provider discovery within huggingface_hub itself.")
    except Exception as e:
        print(f"Error during Mistral chat_completion (direct client): {e}")
        import traceback
        traceback.print_exc()


    # --- Test 2: Text Generation directly with InferenceClient (often a simpler pathway) ---
    print("\n--- Testing InferenceClient.text_generation with mistralai/Mistral-7B-Instruct-v0.1 ---")
    try:
        client_textgen = InferenceClient(model="mistralai/Mistral-7B-Instruct-v0.1", token=token)
        # The text_generation method usually returns a string or an object with generated_text
        response_textgen = client_textgen.text_generation(
            prompt="What is the capital of France?", 
            max_new_tokens=50
        )
        print("Mistral text_generation response (via direct client):", response_textgen)
    except Exception as e:
        print(f"Error during Mistral text_generation (direct client): {e}")
        import traceback
        traceback.print_exc()

    # --- Test 3: Text Generation with a very basic model like gpt2 ---
    print("\n--- Testing InferenceClient.text_generation with gpt2 ---")
    try:
        client_gpt2 = InferenceClient(model="gpt2", token=token)
        response_gpt2 = client_gpt2.text_generation(prompt="Once upon a time", max_new_tokens=20)
        print("gpt2 text_generation response (via direct client):", response_gpt2)
    except Exception as e:
        print(f"Error during gpt2 text_generation (direct client): {e}")
        import traceback
        traceback.print_exc()