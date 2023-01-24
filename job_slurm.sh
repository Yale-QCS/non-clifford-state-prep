#!/bin/bash
#SBATCH --output=slurm-%A-%a.out
#SBATCH --error=slurm-%A-%a.err
#SBATCH --array=2              # array jobs 1-3
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=10       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=2G         # memory per cpu-core (4G is default)
#SBATCH --time=6:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends

module load Julia

echo "Julia module loaded."

julia --project=. -e 'include("test.jl")'

echo "Julia test passed."

julia --project=. -e 'import Pkg; Pkg.instantiate(); include("src/run.jl")'

# for interative testing:
# julia --project=. -e 'import Pkg; Pkg.instantiate(); ENV["SLURM_ARRAY_TASK_ID"]=1; ENV["SLURM_CPUS_PER_TASK"]=1; include("src/run.jl")'

