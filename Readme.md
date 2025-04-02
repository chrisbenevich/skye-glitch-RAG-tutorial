# How to Run RAG Tutorial Code on an HPC

## 1. Retrive the original code and dataset 

```bash
git clone https://github.com/skye-glitch/RAG_tutorial.git
```

## 2. Enter Slurm script 

While you are able to get a compute node with [idev](https://docs.tacc.utexas.edu/software/idev/), you will have to monitor in the Terminal when you connect to the node and then submit your job. For greater productivity, it is recommended to submit your job using Slurm script. This way, you can request the compute node, submit your job and be alerted when it is complete. 

In this step, edit the below Slurm script using your TACC account information [SBATCH](https://tacc.github.io/TeachingWithTACC/02.running_jobs/) and enter it at the command line.

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

## 3. Enter script and submit job to compute node on HPC

Beneath the Slurm script at the command line, enter the code from "rag_ce_example.py" at https://github.com/chrisbenevich-nsalccftaccut-ai-intern/skye-glitch-RAG-tutorial/blob/main/rag_ce_example.py line by line. Begin at the first line, https://github.com/chrisbenevich-nsalccftaccut-ai-intern/skye-glitch-RAG-tutorial/blob/a795b2d32594dd2faaa572d93036ebad14cdb205/rag_ce_example.py#L1 and finish with https://github.com/chrisbenevich-nsalccftaccut-ai-intern/skye-glitch-RAG-tutorial/blob/a795b2d32594dd2faaa572d93036ebad14cdb205/rag_ce_example.py#L123.

## 4. Change directory

Change the directory to where the content resides in your directory on the HPCs. If you cloned the GitHub repo to your home directory to a folder called RAG_tutorial as I did, you will enter the HPC system at your home directory. To then move from your home directory to a folder called RAG_tutorial, enter the following at the command line.

```bash
cd RAG_tutorial
```

## 5. Load the apptainer module with Slurm script

```bash
module load tacc-apptainer
```

## 6. Pull container into your HPC $SCRATCH directory with Slurm script
Run the following once.

```bash
current_dir=$(pwd)
cd $SCRATCH
apptainer pull docker://skyeglitch/taccgptback:latest
cd "$current_dir"
```

## 7. Download model with Slurm script
Run the following once.

```bash
apptainer exec $SCRATCH/taccgptback_latest.sif \
    huggingface-cli download facebook/opt-1.3b --local-dir $SCRATCH/facebook/opt-1.3b/
```

## 8. Launch command in container with Slurm script

```bash
apptainer exec --nv $SCRATCH/taccgptback_latest.sif \
python3 rag_ce_example.py \
--path="$SCRATCH/facebook/opt-1.3b/" \
--MODEL_NAME="$SCRATCH/facebook/opt-1.3b/" 
```

You have now run the tutorial code.

## 9. Instantiated database

While not a separate step, note that after running "rag_ce_example.py." once, you have now loaded, or "instantiated," the database. The code to instantiate the database is located at the following line of code in the .py.

https://github.com/chrisbenevich-nsalccftaccut-ai-intern/skye-glitch-RAG-tutorial/blob/789c1fcc8594d77c4984e7f5be9a7a22134bedc6/rag_ce_example.py#L63  

## 10. Prevent multiple database instantiations

To avoid loading the database more than once, comment out the following line of code in the .py.

https://github.com/chrisbenevich-nsalccftaccut-ai-intern/skye-glitch-RAG-tutorial/blob/789c1fcc8594d77c4984e7f5be9a7a22134bedc6/rag_ce_example.py#L63

## 11. Enable database queries

To query the database consequently, uncomment the following line of code in the .py.

https://github.com/chrisbenevich-nsalccftaccut-ai-intern/skye-glitch-RAG-tutorial/blob/317f544579e16de79e79ef36b3e97be03fd7bbde/rag_ce_example.py#L65

# Understanding RAG Tutorial Components and Dependencies

Once you have seen how the job runs on an HPC, explore the components and dependencies of a RAG by reading the below documentation of "rag_ce_example.py" These explanations of the code were initially provided by prompt engineered chatbot responses, then both lightly edited for brevity and interspersed logically among the code by a human. One way to better understand how a RAG works is to read the description of the function of the code of the first component, read its corresponding code, do the same for the second component and analyze how the two components relate or depend on each other. Following is the documentation.

## 1. Install packages

First, import torch, a library for machine learning tasks. Then, import pretrained models from the HuggingFace transformers library: 

* The pipeline function provides a simple API to perform NLP tasks using pretrained models. It abstracts away the complexity of loading models and tokenizers, making it easy to use them for inference.
* AutoConfig is a class that helps automatically load the configuration for a pretrained model, including details about the model architecture, hyperparameters, and other settings necessary for initializing the model.
* AutoTokenizer is a class in the transformers library that helps automatically load the appropriate tokenizer for a pretrained model, making it easier to switch between different models and their respective tokenizers.
* AutoModelForCausalLM is a class that loads a pretrained model with a causal language modeling head. Causal language models are designed to predict the next token in a sequence. A token is an NLP unit that represents a piece of text.
* GenerationConfig is a class that configures text generation parameters, such as setting the maximum length of the generated text or choosing the decoding strategy. A decoding strategy determines how the model selects the next word in a sequence based on the probabilities assigned to each possible word. 


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

placeholder for next explanatory text

* infer_auto_device_map .
* load_checkpoint_and_dispatch .
* init_empty_weights .

```bash

from accelerate import infer_auto_device_map
from accelerate import load_checkpoint_and_dispatch
from accelerate import init_empty_weights
```

##2. Define and retrieve data

Next, xxxx :   

* xxx

```bash
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def main(in_path="../facebook/opt-1.3b/", in_MODEL_NAME="../facebook/opt-1.3b/"):
    path = in_path
    tokenizer = AutoTokenizer.from_pretrained(path)   
    model_config = AutoConfig.from_pretrained(path)
    
   
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(model_config)
    device_map = infer_auto_device_map(model)
    model = load_checkpoint_and_dispatch(model, path, device_map=device_map)
    model.eval()
    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    model.resize_token_embeddings(len(tokenizer))
```

##3. Load, split, chunk and embed documents

Next, xxxx :   

* xxx

```bash
    # RAG
    loader = DirectoryLoader('./ce311k/notebooks/lectures/', glob="**/*_solutions.md", show_progress=True, loader_cls=UnstructuredMarkdownLoader, load_hidden=False)
    # fix for ssl.SSLCertVerificationError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:1007)
    import ssl
    ssl._create_default_https_context = ssl._create_stdlib_context
    import nltk
    nltk.download('averaged_perceptron_tagger_eng')
    nltk.download('punlt_tab')
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
    texts = text_splitter.split_documents(docs)
    #change to cuda on machine with GPU resource
    embeddings = HuggingFaceEmbeddings(
        model_name="thenlper/gte-large",
        model_kwargs={'device': 'cuda'},
        encode_kwargs={"normalize_embeddings": True},
    )
```

##3. Store documents

Next, xxxx :   

* xxx

```bash
    # only need to insert document once
    db = Chroma.from_documents(texts, embeddings, persist_directory="db_ce")
    # after you run the code for the first time, you can re-use the database with the following command
    # db = Chroma(persist_directory="db_ce", embedding_function=embeddings)
    # you can test retrieved results with the following lines:
    # results = db.similarity_search("data structure", k=2)
    # print(results[0].page_content)
```

##4. Augment documents with a prompt

Next, xxxx :   

* xxx

```bash
       

    messages = ["How to solve system of linear equations using Gauss Elimination?", "How to check length of matrix with python?", "Where can I read more about Numpy?"]
    temperature = 0.8
```

##5. Configure LLM generation settings

Next, xxxx :   

* xxx

```bash
    MODEL_NAME = in_MODEL_NAME
    generation_config = GenerationConfig.from_pretrained(MODEL_NAME)
    generation_config.temperature = temperature
    generation_config.top_p = 0.95
    generation_config.top_k = 40
    generation_config.do_sample = True if temperature > 0.0 else False
    generation_config.max_new_tokens = 512
```

##6. Tune and test hyperparameter changes

Next, tune and test changes to the hyperparameters on the same code block as in the previous step. For example,  :   

* xxx

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

##7. Document generation pipeline

Next, set up a text generation pipeline using Hugging Face's Transformers library and integrate it with LangChain:   

* xxx


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

##8. Use an LLM and a similarity search database to generate answers to questions 

Next, xx:   

* xxx


```bash

    for question in messages:
        results = db.similarity_search(question, k=2)
        retrieved_context = format_docs(results)
        prompt = f"system: You are a TA for students. Use the following retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use one sentences maximum and keep the answer concise. Give students hints instead of telling them the answer.\"\n human:{retrieved_context} \n human: {question}"
        inputs = tokenizer(prompt,return_tensors="pt").to(0)
        outputs = model.generate(**inputs, max_new_tokens=512, do_sample=True)[0]
        print("Here is the answer:")
        print(tokenizer.decode(outputs.cpu().squeeze()).split(question)[-1])
        print("=====================================================")
 ```

##9. Set up command-line interface (CLI) to run the script 

This setup allows you to run the script from the command line, passing the model path and name as arguments:   

* xxx

```bash           

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Put in model name and path')
    parser.add_argument('--path', metavar='path', 
                        help='the path to model')
    parser.add_argument('--MODEL_NAME', metavar='path', 
                        help='name of model')
    args = parser.parse_args()
    main(in_path=args.path, in_MODEL_NAME=args.MODEL_NAME)
```


