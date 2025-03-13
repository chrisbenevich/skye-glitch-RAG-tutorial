# Instruction: How to Run RAG Tutorial Code

## 1. Retrive the Code and Dataset from Github

```bash
git clone https://github.com/skye-glitch/RAG_tutorial.git
```
## 2. Change Directory

change directory into your RAG_tutorial direcotory where the content resides

# insert example command 

## 3. Submit Job

Either: 

submit a job with [SBATCH](https://tacc.github.io/TeachingWithTACC/02.running_jobs/) (after you adapt the slurm script according to your account information on TACC)

```bash
sbatch inference_tutorial.slurm
```

or:

get a compute node with [idev](https://docs.tacc.utexas.edu/software/idev/) command and run the commands in the inference_tutorial.slurm script line by line.

## 4. Load the apptainer module with Slurm script

```bash
module load tacc-apptainer
```

### 5. Pull container into your $SCRATCH directory with Slurm script
You only need to run this ONCE,
comment out these lines after the first run. 

```bash
current_dir=$(pwd)
cd $SCRATCH
# CB edit below as :latest
apptainer pull docker://skyeglitch/taccgptback_latest
cd "$current_dir"
```

### 6. Download model with Slurm script
You only need to run this ONCE,
comment out these lines after the first run.

```bash
apptainer exec $SCRATCH/taccgptback_latest.sif \
    huggingface-cli download facebook/opt-1.3b --local-dir $SCRATCH/facebook/opt-1.3b/
```
### 7. Launch command in container with Slurm script

```bash
apptainer exec --nv $SCRATCH/taccgptback_latest.sif \
python3 rag_ce_example.py \
--path="$SCRATCH/facebook/opt-1.3b/" \
--MODEL_NAME="$SCRATCH/facebook/opt-1.3b/" 
```

## 8. Python Script Highlights

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

