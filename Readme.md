# How to Run RAG Tutorial Code on an HPC


## 1. Understand basic Terminal use 

This tutorial assumes a basic understanding of the Terminal. If needed, one introduction to using command line prompts in the Terminal is at the following link.

https://tacc-reproducible-intro-hpc.readthedocs.io/en/latest/intro_to_command_line/overview.html 


## 2. SSH on to the HPC  

At the command line, enter your TACC email, password and MFA token. MFA details are at the following link.

https://tap.tacc.utexas.edu/mfalogin/


## 3. Retrive the original code and dataset 

If you do not have a GitHub account, open one. Clone the repository with the following command. 

```bash
git clone https://github.com/skye-glitch/RAG_tutorial.git
```

For detailed instructions on cloning GitHub repositories, visit the following link. 

https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository 

## 4. Load modules

Based on which HPC you are on, verify the version of Python available. You will need Python xxx and Cuda xxx at minimum. To use Python, load the module for the Python package by entering the following command.

```bash

module load python/3.8.5
module load cuda/11.2
```


## 5. Set up virtual environment

To ensure the correct packages are available for your project, create a virtual environment. 

* For business continuity and organizational purposes, consider where you would like to save the virtual environment. In this example, a new directory, called RAG_tutorial, and a virtual environment, called RAG_VE, are created at the command prompt as follows.

```bash
mkdir RAG_tutorial
python3 -m venv RAG_VE
```

* Verify that your directory installed, change directories and verify that your virtual environment installed.

```bash
ls
cd RAG_VE/
ls
```

* Activate the virtual environment.

```bash
source RAG_VE/bin/activate
```

* Install the PyTorch package in the virtual environment.

```bash
pip install torch langchain accelerate
```


* If for any reason you need to end your session and return to the tutorial, you can load the virtual environment from the beginning of your session as follows.

```bash
cd RAG_VE/
ls
```

* Once you return to the tutorial and reload your virtual environment, proceed to the next step in the tutorial.


## 6. Enter Slurm script 

In this step, request a compute node by editing the below Slurm script using your TACC account information [SBATCH](https://tacc.github.io/TeachingWithTACC/02.running_jobs/).


```bash
#!/bin/bash
#SBATCH -A your_allocation
#SBATCH --time=00:05:00
#SBATCH -o RAG-%J.o
#SBATCH -e RAG-%J.e
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p rtx
#SBATCH --mail-user=your_email
#SBATCH --mail-type=all
```


## 7. Enter script and submit job to compute node on HPC

Modify the environment to operate in Python and request to run the code.

```bash
python3 rag_ce_example.py 
```

You have now submitted your job. Once a node becomes available and your code runs, you will be alerted. 


## 8. Instantiated database

While not a separate step, note that after running "rag_ce_example.py." once, you have now loaded, or "instantiated," the database. The code to instantiate the database is located at the following line of code in the .py.

https://github.com/chrisbenevich-nsalccftaccut-ai-intern/skye-glitch-RAG-tutorial/blob/789c1fcc8594d77c4984e7f5be9a7a22134bedc6/rag_ce_example.py#L63  


## 9. Prevent multiple database instantiations

To avoid loading the database more than once, comment out the following.

https://github.com/chrisbenevich-nsalccftaccut-ai-intern/skye-glitch-RAG-tutorial/blob/789c1fcc8594d77c4984e7f5be9a7a22134bedc6/rag_ce_example.py#L63


## 10. Enable database queries

To query the database consequently, uncomment the following.

https://github.com/chrisbenevich-nsalccftaccut-ai-intern/skye-glitch-RAG-tutorial/blob/317f544579e16de79e79ef36b3e97be03fd7bbde/rag_ce_example.py#L65


## 11. Test retrieved results

To test retrieved results, uncomment the following lines.

https://github.com/chrisbenevich-nsalccftaccut-ai-intern/skye-glitch-RAG-tutorial/blob/9c2344a7eae9917c2c66400574ae3a777630a56d/rag_ce_example.py#L67

https://github.com/chrisbenevich-nsalccftaccut-ai-intern/skye-glitch-RAG-tutorial/blob/9c2344a7eae9917c2c66400574ae3a777630a56d/rag_ce_example.py#L68


# Understanding RAG Tutorial Components and Dependencies

Once you have seen how the job runs on an HPC, explore the components and dependencies of a RAG by reading the below documentation of "rag_ce_example.py" at https://github.com/chrisbenevich-nsalccftaccut-ai-intern/skye-glitch-RAG-tutorial/blob/main/rag_ce_example.py. These explanations of the code were initially provided by prompt engineered chatbot responses, then both lightly edited for brevity and interspersed logically among the code by a human. 

One way to better understand how a RAG works is to read the description of the function of the code of the first component, read its corresponding code, do the same for the second component and analyze how the two components relate or depend on each other. Following is the documentation of ten steps to set up a RAG.

## 1. Install packages

First, import torch, a library for machine learning tasks. Then, import pretrained models from the HuggingFace transformers library. 

```bash
import torch

from transformers import pipeline
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, GenerationConfig
```

Next, install the LangChain framework and Hugging Face's models; they integrate to provide open-source libraries and tools for building context awareness and reasoning:   

* HuggingFacePipeline provides a high-level API to perform tasks such as text classification and question answering.
* UnstructuredMarkdownLoader loads markdown files.
* DirectoryLoader loads documents from a directory on your filesystem.
* Hugging Face Transformers library in Python provides a wide range of pre-trained models for NLP tasks, such as text classification, named entity recognition and question answering.
* RecursiveCharacterTextSplitter splits text into smaller chunks by recursively looking at characters.
* HuggingFaceEmbeddings is a class in LangChain that utilizes Hugging Face's sentence transformer models to generate embeddings for text. Embeddings are numerical representations of text that capture semantic meaning.
* Chroma is a vector database that stores and manages embeddings.
* StrOutputParser parses LLMs output into the most likely string.
* EmbeddingsRedundantFilter compares embeddings to filter out redundant documents.
* DocumentCompressorPipeline compresses documents, retaining only the information relevant to the query.

```bash
# for RAG
from langchain_huggingface import HuggingFacePipeline
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.document_loaders import DirectoryLoader
import transformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma 
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
```


## 2. Prepare and load the model

Next, optimize and manage the deployment of the model across various devices:

* Infer the optimal device map for distributing model layers across available devices (e.g., CPUs, GPUs) to help utilize hardware resources efficiently.
* Based on the inferred device map, load a model checkpoint and dispatche the model layers to the appropriate devices.
* Initialize a model with empty weights, useful for setting up the model structure before loading actual weights.

```bash

from accelerate import infer_auto_device_map
from accelerate import load_checkpoint_and_dispatch
from accelerate import init_empty_weights
```

Next, to prepare the model for inference (i.e., making predictions based on a trained model), load it, distribute it across devices and configure its parameters:   
 
* Take a list of documents and format them by joining their content with double newline characters to create a blank line between text segments.

```bash
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
``` 

* Set the path to the model.
* Load the tokenizer from the specified path using AutoTokenizer.
* Load the model configuration using AutoConfig.

```bash
def main(in_path="../facebook/opt-1.3b/", in_MODEL_NAME="../facebook/opt-1.3b/"):
    path = in_path
    tokenizer = AutoTokenizer.from_pretrained(path)   
    model_config = AutoConfig.from_pretrained(path)
``` 
 
* Initialize the model with empty weights.
* Create the model using the configuration.

```bash
with init_empty_weights():
        model = AutoModelForCausalLM.from_config(model_config)
``` 
  
* Infer the device map for distributing the model across available devices.
* Load the model checkpoint and dispatch it across the devices.

```bash
device_map = infer_auto_device_map(model)
    model = load_checkpoint_and_dispatch(model, path, device_map=device_map)
``` 

* Set the model to evaluation mode.
* Configure the end and padding token IDs.
* Resize the token embeddings to match the tokenizer's vocabulary size.

```bash
model.eval()
    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    model.resize_token_embeddings(len(tokenizer))
```


## 3. Load, split, chunk and embed documents

Next, set up a pipeline to load Markdown files, fix SSL issues, download necessary NLTK data, split the text into manageable chunks, and generate embeddings using a specified model:   

* Initialize a DirectoryLoader to load markdown files from the specified directory (./ce311k/notebooks/lectures/). The DirectoryLoader uses a glob pattern to match files ending with _solutions.md, shows progress during loading, uses UnstructuredMarkdownLoader to process the files, and does not load hidden files.
* The next line of code fixes an SSL certificate verification error by changing the default HTTPS context. 

```bash
    # RAG
    loader = DirectoryLoader('./ce311k/notebooks/lectures/', glob="**/*_solutions.md", show_progress=True, loader_cls=UnstructuredMarkdownLoader, load_hidden=False)
    # fix for ssl.SSLCertVerificationError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:1007)
    import ssl
    ssl._create_default_https_context = ssl._create_stdlib_context
```

* Download specific NLTK data packages needed for text processing.
* Load the documents from the specified directory.

```bash

    import nltk
    nltk.download('averaged_perceptron_tagger_eng')
    nltk.download('punlt_tab')
    docs = loader.load()
```

* Initialize a RecursiveCharacterTextSplitter to split the loaded documents into chunks of 1024 characters with an overlap of 64 characters.
* Initialize HuggingFaceEmbeddings using the specified model (thenlper/gte-large) and set it to use CUDA for GPU acceleration. Normalize the embeddings.

```bash
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
    texts = text_splitter.split_documents(docs)
    #change to cuda on machine with GPU resource
    embeddings = HuggingFaceEmbeddings(
        model_name="thenlper/gte-large",
        model_kwargs={'device': 'cuda'},
        encode_kwargs={"normalize_embeddings": True},
    )
```


## 4. Store documents

Instantiate the database.   

```bash
    db = Chroma.from_documents(texts, embeddings, persist_directory="db_ce")
```

Note that after running "rag_ce_example.py." once, you have now loaded, or "instantiated," the database. To avoid loading the database more than once, comment out the above line of code. 

To query the database consequently, uncomment the following.

```bash
    # db = Chroma(persist_directory="db_ce", embedding_function=embeddings)
```

To test retrieved results, uncomment the following.

```bash
    # results = db.similarity_search("data structure", k=2)
    # print(results[0].page_content)
```


## 5. Initialize questions and set temperature

Next, include the queries for the model and the temperature for the results. A control of the randomness of the model's predictions, a temperature of 0.8 indicates a moderate level of randomness, allowing for more diverse outputs.   

```bash
       
    messages = ["How to solve system of linear equations using Gauss Elimination?", "How to check length of matrix with python?", "Where can I read more about Numpy?"]
    temperature = 0.8
```


## 6 Configure LLM generation settings

Next, configure the generation settings (a subset of hyperparameters specifically tailored for the text generation phase) for the LLM with Hugging Face's Transformers library. This helps control the behavior of the model during text generation, balancing between randomness and coherence:

* Temperature controls the randomness of predictions by scaling the logits before applying softmax. Lower values make the model more deterministic, while higher values make it more random.
* top_p is used for nucleus sampling and sets a probability threshold for selecting the next token. 
* top_k limits the sampling pool to the top k tokens with the highest probabilities, helping control the diversity of the generated text.

```bash
    MODEL_NAME = in_MODEL_NAME
    generation_config = GenerationConfig.from_pretrained(MODEL_NAME)
    generation_config.temperature = temperature
    generation_config.top_p = 0.95
    generation_config.top_k = 40
```

* do_sample, a boolean flag, determines whether to sample from the distribution or take the most probable token. It's set to True if temperature is greater than 0, allowing for more diverse outputs.
* max_new_tokens sets the maximum number of new tokens to generate in the output sequence.

```bash
    generation_config.do_sample = True if temperature > 0.0 else False
    generation_config.max_new_tokens = 512
```

While not a separate step of building a RAG, to better understand what hyperparameters do, take another look at the previous block of code. Experiment with changing one hyperparameter at a time. Insert a comment documenting what you changed and what you expect as a result of the change. Then, re-run the entire code and document the actual result. An example of documenting qualitative changes to hyperparameters follows:   

```bash
    MODEL_NAME = in_MODEL_NAME
    generation_config = GenerationConfig.from_pretrained(MODEL_NAME)
    generation_config.temperature = temperature
    # To test hyperparamter, changed p from 0.95 to 0.75, anticipating loss in accuracy
    # Result: last section of output was nonsensical
    generation_config.top_p = 0.95
    # To test hyperparamter, changed k from 40 to 5, anticipating increase in accuracy
    # but perhaps fewer sentences
    # Result: total output was shorter; unclear effect on accuracy but appeared to 
    # include human-to-human conversation not present in previous runs
    generation_config.top_k = 40
    generation_config.do_sample = True if temperature > 0.0 else False
    # To test hyperparameter, changed tokens from 512 to 100, anticipating 
    # nonsensical output. Resulted in: incorrect use of $ symbol in mathematical equation
    # and repeating an input question except with slightly different word order
    generation_config.max_new_tokens = 512
```


## 7. Document generation pipeline

Next, set up a text generation pipeline using Hugging Face's Transformers library and integrate it with LangChain:   

* text_pipeline specifies the task as "text-generation" and uses the provided model, tokenizer, and generation_config. 
* torch_dtype=torch.float16 sets the data type for the model's tensors to float16, helping reduce memory usage and speed up computations.
* llm, a variable and instance of HuggingFacePipeline, effectively wraps the pipeline in a way that can be used for generating text based on the model and configuration provided.

```bash
    
    text_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        generation_config=generation_config,
    )

    llm = HuggingFacePipeline(pipeline=text_pipeline) 
```


## 8. Augment the prompt and generate answers to questions 

Retrieve relevant documents, format them, provide context to augment the prompt and generate a concise answer. For each question in the messages list, the code performs the following steps:   

* Search the database for the top 2 documents most similar to the question.
* Format the retrieved documents into a suitable context for the model.
* Instruct the model and the retrieved context with the prompt variable.

```bash

    for question in messages:
        results = db.similarity_search(question, k=2)
        retrieved_context = format_docs(results)
        prompt = f"system: You are a TA for students. Use the following retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use one sentences maximum and keep the answer concise. Give students hints instead of telling them the answer.\"\n human:{retrieved_context} \n human: {question}"
```

* Tokenize the prompt and prepare it for input to the model, moving the tensors to the appropriate device (e.g., GPU).
* Generate output from the model based on the prompt, with a maximum of 512 new tokens and sampling enabled.
* Decode and print the answer, extracting it from the decoded text by splitting it at the question and taking the part after it.

```bash

        inputs = tokenizer(prompt,return_tensors="pt").to(0)
        outputs = model.generate(**inputs, max_new_tokens=512, do_sample=True)[0]
        print("Here is the answer:")
        print(tokenizer.decode(outputs.cpu().squeeze()).split(question)[-1])
        print("=====================================================")
 ```


Congratulations! You have now recreated and run a RAG on an HPC.
