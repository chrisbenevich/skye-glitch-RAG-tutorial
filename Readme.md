# Instruction: How to Run RAG Turorial Code


## 1. Retrive the Code and Dataset from Github

```bash
git clone https://github.com/skye-glitch/RAG_tutorial.git
```

## 2. Slurm Script Step-by-Step

### 2.1 Load the apptainer module

```bash
module load tacc-apptainer
```

### 2.2 Pull container into your $SCRATCH directory
You only need to run this ONCE,
comment out these lines after the first run. 

```bash
current_dir=$(pwd)
cd $SCRATCH
apptainer pull docker://skyeglitch/taccgptback_latest
cd "$current_dir"
```

### 2.3 Download model
You only need to run this ONCE,
comment out these lines after the first run.

```bash
apptainer exec $SCRATCH/taccgptback_latest.sif \
    huggingface-cli download facebook/opt-1.3b --local-dir $SCRATCH/facebook/opt-1.3b/
```
### 2.4 Launch command in container

```bash
apptainer exec --nv $SCRATCH/taccgptback_latest.sif \
python3 rag_ce_example.py \
--path="$SCRATCH/facebook/opt-1.3b/" \
--MODEL_NAME="$SCRATCH/facebook/opt-1.3b/" 
```

## 3. Python Script Highlights

https://github.com/skye-glitch/RAG_tutorial/blob/main/rag_ce_example.py#L62
```python
db = Chroma.from_documents(texts, embeddings, persist_directory="db_ce")
```
This operation to insert the document into your database only need to be run once.
After first run, you can use 
https://github.com/skye-glitch/RAG_tutorial/blob/main/rag_ce_example.py#L64

```python
db = Chroma(persist_directory="db_ce", embedding_function=embeddings)
```
to inquire the databse.

