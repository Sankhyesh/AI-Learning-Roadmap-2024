# LangChain RAG-based Applications and Document Loaders

- The content introduces a shift in focus within a LangChain learning playlist, moving towards building **RAG (Retrieval Augmented Generation)** based applications. The memory component, previously part of LangChain, is now being integrated into LangGraph and will be covered later.
 
- **Recap of Playlist Progress:**

  - The playlist has so far covered **LangChain's important components** (Models, Prompts, Chains) with hands-on coding.
  - It has also focused on **LangChain's core concepts** (e.g., LCEL - LangChain Expression Language) in-depth to build strong fundamentals.
  - The goal was to make learners ready to build any LLM-based application using LangChain.

- **Introduction to RAG (Retrieval Augmented Generation):**

  - **RAG** is identified as a significant use case for generative AI, especially for chatbots.
  - **Problem RAG Solves:** Standard LLMs like ChatGPT may not have access to:
    - Current, real-time information (due to training data cutoffs).
    - Private or personal data (e.g., personal emails, company-specific documentation).
  - **RAG's Solution:** It connects an LLM to an **external knowledge base** (e.g., company databases, PDFs, personal documents).
    - When a user asks a question the LLM doesn't know, the LLM queries this knowledge base for relevant information.
    - The LLM then uses this retrieved information as **context** to generate an accurate and grounded response.
  - **Definition of RAG:** A technique combining **information retrieval** (from the external knowledge base) with **language generation** (by the LLM). The model retrieves relevant documents and uses them as context.
  - **Benefits of RAG:**
    - Access to **up-to-date information**.
    - Enhanced **privacy** (data isn't necessarily uploaded to a third-party LLM provider if managed locally).
    - Overcomes limitations of **document size** and context length by processing documents in manageable parts (chunks).
  - RAG-based applications are a powerful trend in the industry.

- **Learning Plan for RAG in LangChain:**

  - Instead of building a full RAG application at once, the approach will be to first learn its **important components**.
  - The key components of RAG applications are:
    1.  **Document Loaders**
    2.  **Text Splitters**
    3.  **Vector Databases**
    4.  **Retrievers**
  - This video focuses on the first component: **Document Loaders**.

- **Document Loaders in LangChain:**

  - **Purpose:** Components used to load data from various sources into a **standardized format** within LangChain.
  - **Standardized Format:** Data is typically loaded as **Document objects**. This common format allows downstream components (like chunking, embedding, retrieval, generation) to work consistently with the data.
  - **Document Object Structure:** Each **Document object** generally contains:
    - **`page_content`**: The actual content/text from the source.
    - **`metadata`**: Information about the data, such as its source (e.g., file name, URL), creation date, page number, etc.
  - LangChain offers hundreds of document loaders, but the video will focus on the core principles and the four most commonly used ones.
  - The fundamental working principle is similar across most document loaders, so understanding one helps in using others.
  - All document loaders are found in the `langchain_community.document_loaders` package.
  - The typical process involves:
    1.  Creating a `loader` object for the specific document type, providing necessary parameters (e.g., file path).
    2.  Calling the `loader.load()` method.
    3.  This method returns a **list of Document objects**.

- **Specific Document Loaders Covered:**

  1.  **TextLoader:**

      - **Function:** Loads text files (`.txt`) into Document objects.
      - **Usage:** Instantiate `TextLoader` with the `file_path` and optional `encoding` (e.g., `utf-8` for special characters).
      - The `load()` method typically returns a list containing a single Document object for a text file, where `page_content` is the entire text and `metadata` includes the source file path.
      - This loaded content can then be passed to an LLM chain for tasks like summarization.

      Example code:

      ```python
      from langchain_community.document_loaders import TextLoader
      from langchain_openai import ChatOpenAI
      from langchain_core.output_parsers import StrOutputParser
      from langchain_core.prompts import PromptTemplate
      from dotenv import load_dotenv

      load_dotenv()

      # Initialize the loader
      loader = TextLoader('cricket.txt', encoding='utf-8')

      # Load documents
      docs = loader.load()

      # Check the type and number of documents
      print(type(docs))  # List of Document objects
      print(len(docs))   # Number of documents

      # Access content and metadata
      print(docs[0].page_content)  # Content of the first document
      print(docs[0].metadata)      # Metadata of the first document

      # Example of using the loaded content with an LLM chain
      model = ChatOpenAI()
      prompt = PromptTemplate(
          template='Write a summary for the following poem - \n {poem}',
          input_variables=['poem']
      )
      parser = StrOutputParser()
      chain = prompt | model | parser
      print(chain.invoke({'poem':docs[0].page_content}))
      ```

  2.  **PyPDFLoader:**

      - **Function:** Loads PDF files, converting each page into a separate Document object.
      - **Significance:** Highly used for PDF processing. If a PDF has 25 pages, `load()` will return a list of 25 Document objects.
      - **Metadata:** Each Document object's metadata includes the `source` (PDF filename) and the `page` number.
      - **Underlying Library:** Uses the `pypdf` library. Thus, it works best with text-based PDFs and may not perform well with scanned or complex image-based PDFs.
      - Requires `pip install pypdf`.
      - **Other PDF Loaders:** For more complex scenarios (tables, scanned images), other loaders exist:
        - `PDFPlumberLoader`: For tables.
        - `UnstructuredPDFLoader`, `AmazonTextractPDFLoader`, `PyPDFium2Loader`, `PyMuPDFLoader`: For scanned images or more complex structures. Links to LangChain documentation are provided for further exploration.

      Example code:

      ```python
      from langchain_community.document_loaders import PyPDFLoader

      # Initialize the loader with a PDF file
      loader = PyPDFLoader('dl-curriculum.pdf')

      # Load the documents
      docs = loader.load()

      # Print the number of documents (pages)
      print(len(docs))

      # Access content and metadata of specific pages
      print(docs[0].page_content)  # Content of first page
      print(docs[1].metadata)      # Metadata of second page
      ```

  3.  **DirectoryLoader:**

      - **Function:** Loads all files from a specified directory that match a given pattern, using another specified loader for each file.
      - **Use Case:** Useful for loading multiple files (e.g., all PDFs or all `.txt` files) from a folder at once.
      - **Parameters:**
        - `path`: The path to the directory.
        - `glob`: A pattern to match files (e.g., `"*.pdf"` for all PDFs, `"**/[!.]*.txt"` for all text files in all subdirectories, excluding hidden ones).
        - `loader_cls`: The class of the loader to use for the matched files (e.g., `PyPDFLoader` for PDFs).
      - The `load()` method returns a list containing Document objects from all pages of all loaded files. For example, three PDFs with 326, 392, and 468 pages respectively would result in 1186 Document objects in total.

      Example code:

      ```python
      from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

      # Initialize the directory loader
      loader = DirectoryLoader(
          path='books',            # Directory containing the files
          glob='*.pdf',            # Pattern to match PDF files
          loader_cls=PyPDFLoader   # Use PyPDFLoader for each matched file
      )

      # Use lazy_load() for memory efficiency
      docs = loader.lazy_load()

      # Process documents one at a time
      for document in docs:
          print(document.metadata)
      ```

  4.  **Lazy Loading (`.lazy_load()` vs `.load()`):**

      - **Problem with `.load()` (Eager Loading):** Loads all documents into memory at once. This can be time-consuming and memory-intensive for a large number of or very large files.
      - **Solution: `.lazy_load()` (Lazy Loading):**
        - Loads documents on demand, one at a time.
        - Returns a **generator** of Document objects instead of a list.
        - **Benefits:** More memory-efficient, faster initial response as it doesn't wait to load everything. Suitable for processing a large number of documents or for stream processing.
        - Allows iterating through documents one by one, processing each, and then discarding it from memory before loading the next.

  5.  **WebBaseLoader:**

      - **Function:** Loads and extracts text content from web pages.
      - **Underlying Libraries:** Uses `requests` (for HTTP requests) and `Beautiful Soup` (for parsing HTML and extracting text).
      - **Usage:** Instantiate `WebBaseLoader` with a URL (or a list of URLs).
      - `load()` returns a list of Document objects, typically one per URL.
      - **Limitations:** Works best with static web pages (HTML-heavy). May not perform well with JavaScript-heavy dynamic pages. For dynamic pages, `SeleniumURLLoader` is suggested.
      - A project idea is mentioned: creating a Chrome plugin for real-time Q&A with the content of the currently open web page.

      Example code:

      ```python
      from langchain_community.document_loaders import WebBaseLoader
      from langchain_openai import ChatOpenAI
      from langchain_core.output_parsers import StrOutputParser
      from langchain_core.prompts import PromptTemplate
      from dotenv import load_dotenv

      load_dotenv()

      # Initialize model and prompt
      model = ChatOpenAI()
      prompt = PromptTemplate(
          template='Answer the following question \n {question} from the following text - \n {text}',
          input_variables=['question','text']
      )
      parser = StrOutputParser()

      # Initialize and use the web loader
      url = 'https://www.flipkart.com/apple-macbook-air-m2-16-gb-256-gb-ssd-macos-sequoia-mc7x4hn-a/p/itmdc5308fa78421'
      loader = WebBaseLoader(url)
      docs = loader.load()

      # Create and use a chain for question answering
      chain = prompt | model | parser
      print(chain.invoke({
          'question': 'What is the product that we are talking about?',
          'text': docs[0].page_content
      }))
      ```

  6.  **CSVLoader:**

      - **Function:** Loads data from CSV files.
      - **Behavior:** Converts each **row** of the CSV into a separate Document object.
      - **`page_content`:** A string representation of the row, typically "column_name: value, column_name: value, ...".
      - **`metadata`:** Includes the `source` (CSV filename) and the `row` number.
      - Can also be used with `lazy_load()` for large CSV files.

      Example code:

      ```python
      from langchain_community.document_loaders import CSVLoader

      # Initialize the CSV loader
      loader = CSVLoader(file_path='Social_Network_Ads.csv')

      # Load all documents
      docs = loader.load()

      # Print number of documents (rows) and example document
      print(len(docs))            # Number of rows
      print(docs[1])              # Second row as Document object
      ```

- **Overview of Other Document Loaders:**

  - A link to the LangChain documentation is provided, showcasing a wide array of available document loaders categorized by source type (web pages, PDFs, cloud services like S3/Azure/Google Drive, social platforms, messaging services, productivity tools like GitHub, common file types like JSON).
  - The advice is to learn them on a project-by-project basis as needed, rather than trying to learn all of them.

- **Custom Document Loaders:**

  - **Necessity:** If a data source doesn't have a pre-built LangChain document loader.
  - **Capability:** LangChain allows users to create their own **custom document loaders**.
  - **Process:** Involves creating a class that inherits from the `BaseLoader` class and implementing the `load()` and/or `lazy_load()` methods with custom logic for fetching and processing data into Document objects.
  - Many existing loaders in `langchain_community` were developed by the community, highlighting this extensibility. 

[End of Notes]
