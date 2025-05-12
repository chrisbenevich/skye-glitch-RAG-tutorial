import torch


from transformers import pipeline
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, GenerationConfig


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
from accelerate import infer_auto_device_map
from accelerate import load_checkpoint_and_dispatch
from accelerate import init_empty_weights




def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# 5/7/25 change initial ../facebook/opt-1.3b/ path to absolute
def main(in_path="/scratch1/10513/chrisbenevich/RAG_tutorial/facebook/opt-1.3b/", in_MODEL_NAME="/scratch1/10513/chrisbenevich/RAG_tutorial/facebook/opt-1.3b/"):
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

    # RAG
    # 5/7/25 change initial ./ce311k location to absolute path
    loader = DirectoryLoader('/scratch1/10513/chrisbenevich/RAG_tutorial/ce311k/notebooks/lectures/', glob="**/*_solutions.md", show_progress=True, loader_cls=UnstructuredMarkdownLoader, load_hidden=False)
    # fix for ssl.SSLCertVerificationError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:1007)
    import ssl
    ssl._create_default_https_context = ssl._create_stdlib_context
    import nltk
    nltk.download('averaged_perceptron_tagger_eng')
    #4/28/25 edit punlt to punkt, run, see if still throws error
    nltk.download('punkt_tab')
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
    texts = text_splitter.split_documents(docs)
    #change to cuda on machine with GPU resource
    embeddings = HuggingFaceEmbeddings(
        model_name="thenlper/gte-large",
        model_kwargs={'device': 'cuda'},
        encode_kwargs={"normalize_embeddings": True},
    )

    # only need to insert document once
    db = Chroma.from_documents(texts, embeddings, persist_directory="db_ce")
    # after you run the code for the first time, you can re-use the database with the following command
    # db = Chroma(persist_directory="db_ce", embedding_function=embeddings)
    # you can test retrieved results with the following lines:
    # results = db.similarity_search("data structure", k=2)
    # print(results[0].page_content)

       

    messages = ["How to solve system of linear equations using Gauss Elimination?", "How to check length of matrix with python?", "Where can I read more about Numpy?"]
    temperature = 0.8

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

    
    text_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        generation_config=generation_config,
    )

    llm = HuggingFacePipeline(pipeline=text_pipeline) 


    for question in messages:
        results = db.similarity_search(question, k=2)
        retrieved_context = format_docs(results)
        prompt = f"system: You are a TA for students. Use the following retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use one sentences maximum and keep the answer concise. Give students hints instead of telling them the answer.\"\n human:{retrieved_context} \n human: {question}"
        inputs = tokenizer(prompt,return_tensors="pt").to(0)
        outputs = model.generate(**inputs, max_new_tokens=512, do_sample=True)[0]
        print("Here is the answer:")
        print(tokenizer.decode(outputs.cpu().squeeze()).split(question)[-1])
        print("=====================================================")
            
# comment out lines of parser then add the "main" function call as a test
if __name__=="__main__":
#    import argparse
#    parser = argparse.ArgumentParser(description='Put in model name and path')
#    parser.add_argument('--path', metavar='path', 
#                        help='the path to model')
#    parser.add_argument('--MODEL_NAME', metavar='path', 
#                        help='name of model')
#    args = parser.parse_args()
#    main(in_path=args.path, in_MODEL_NAME=args.MODEL_NAME)
    main()
