"# Chef-Companion-Bot" 
## README

# LLM Chatbot for Recipe Information

This project creates a conversational chatbot using a large language model (LLM) to interact with a dataset of Indian recipes. The chatbot can answer questions about recipes, ingredients, preparation times, and instructions. This implementation uses the Mistral-7B-Instruct-v0.1 model for the language model and integrates various tools for data handling, embeddings, and serving the chatbot via a web interface.

## Table of Contents
- [Requirements](#requirements)
- [Setup](#setup)
- [Usage](#usage)
- [Project Structure](#project-structure)


## Requirements
- Google Colab or local environment with GPU support
- Python 3.7+
- Required Python packages listed in `requirements.txt`

## Setup

### Clone the Repository
```bash
git clone https://github.com/your-repo/llm-chatbot-recipe.git
cd llm-chatbot-recipe
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Load the Dataset
Ensure your dataset is available in the path `/content/drive/MyDrive/recipe generation/Cleaned_Indian_Food_Dataset.csv`. You can modify the path in the script to point to your dataset location.

### Prepare the Model
This project uses the `mistralai/Mistral-7B-Instruct-v0.1` model, loaded with quantization configurations to optimize performance. You need to install and load the model using the script.

### Configure the Pipeline
The pipeline configuration is saved and loaded from `/content/drive/MyDrive/llm_chatbot/pipeline_config.json`. This configuration specifies the parameters for the text generation task.

## Usage

### Running the Chatbot

1. **Mount Google Drive:**
   If using Google Colab, start by mounting your Google Drive to access the dataset and save configurations.
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. **Load and Prepare Data:**
   Load the dataset and prepare it by concatenating the recipe information into a single text block, then split it into chunks for embeddings.
   ```python
   import pandas as pd
   df = pd.read_csv('/content/drive/MyDrive/recipe generation/Cleaned_Indian_Food_Dataset.csv')
   # Process the data as shown in the script
   ```

3. **Initialize the Model:**
   Load the pre-trained model and tokenizer.
   ```python
   from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
   model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb_config)
   tokenizer = AutoTokenizer.from_pretrained(model_path)
   ```

4. **Create the Conversational Retrieval Chain:**
   Initialize the embeddings and create the retrieval chain using FAISS.
   ```python
   from langchain.vectorstores import FAISS
   db = FAISS.from_documents(chunked_docs, embeddings)
   retriever = db.as_retriever(search_type="similarity")
   qa_chain = ConversationalRetrievalChain.from_llm(mistral_llm, retriever, return_source_documents=True)
   ```

5. **Serve the Chatbot:**
   Start the FastAPI server and expose it via ngrok for public access.
   ```python
   import uvicorn
   uvicorn.run(app, port=8000)
   ```

### Web Interface
Access the chatbot via the web interface provided by the FastAPI server. The HTML file included in the script sets up a simple chat UI.


---

This README provides an overview of the setup and usage of the LLM chatbot for recipe information. It covers the installation of dependencies, preparation of the dataset, model initialization, and serving the chatbot via a web interface.
