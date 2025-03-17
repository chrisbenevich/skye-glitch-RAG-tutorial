# How to Run RAG Tutorial Code

## 1. Retrive the original code and dataset 

```bash
git clone https://github.com/skye-glitch/RAG_tutorial.git
```

## 2. Submit job to compute node on HPC

Either: 

edit the below Slurm script using your TACC account information [SBATCH](https://tacc.github.io/TeachingWithTACC/02.running_jobs/) and submit the job by executing the below one line at a time at the command line.

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

or:

get a compute node with [idev](https://docs.tacc.utexas.edu/software/idev/) 

```bash
idev -p rtx
```

## 3. Change directory

Change directory to where the content resides. To move from the home directory to a folder called RAG_tutorial:

```bash
cd RAG_tutorial
```

## 4. Load the apptainer module with Slurm script

```bash
module load tacc-apptainer
```

## 5. Pull container into your HPC $SCRATCH directory with Slurm script
Run the following once:

```bash
current_dir=$(pwd)
cd $SCRATCH
apptainer pull docker://skyeglitch/taccgptback:latest
cd "$current_dir"
```
Then comment out the above lines in "inference_tutorial.slurm."

## 6. Download model with Slurm script
Run the following once:

```bash
apptainer exec $SCRATCH/taccgptback_latest.sif \
    huggingface-cli download facebook/opt-1.3b --local-dir $SCRATCH/facebook/opt-1.3b/
```

Then comment out the above lines in "inference_tutorial.slurm."

## 7. Launch command in container with Slurm script

```bash
apptainer exec --nv $SCRATCH/taccgptback_latest.sif \
python3 rag_ce_example.py \
--path="$SCRATCH/facebook/opt-1.3b/" \
--MODEL_NAME="$SCRATCH/facebook/opt-1.3b/" 
```

## 8. Instantiate database

After running "rag_ce_example.py." once, comment out the database line of code located at 

https://github.com/chrisbenevich-nsalccftaccut-ai-intern/skye-glitch-RAG-tutorial/blob/789c1fcc8594d77c4984e7f5be9a7a22134bedc6/rag_ce_example.py#L63  

## 9. Enable database queries

To query the database consequently, uncomment the line of code located at

https://github.com/chrisbenevich-nsalccftaccut-ai-intern/skye-glitch-RAG-tutorial/blob/317f544579e16de79e79ef36b3e97be03fd7bbde/rag_ce_example.py#L65
