# Fault-tolerant non-clifford gates using error detection
## Manuscript

Source code for paper: [Fault Tolerant Non-Clifford State Preparation for Arbitrary Rotations](https://arxiv.org/abs/2303.17380)

Credit note: Part of the source code is built upon software from
[Virtualized Logical Qubits: A 2.5D Architecture for Error-Corrected Quantum Computing](https://arxiv.org/abs/2009.01982), from [cduck/VLQ](https://github.com/cduck/VLQ)


## Install

1. Clone this repository
    ```bash
    git clone https://github.com/yongshanding/vlq
    cd vlq
    ```

2. Install Julia (tested with 1.4.2): [julialang.org/downloads](https://julialang.org/downloads/)

3. Set path environment. In `~/.bashrc`:
    ```bash
    export PATH="$PATH:/path/to/<Julia directory>/bin"
    ```
    
3. Install required Julia packages (run from the `vlq/` directory) from the julia REPL:
    ```julia
    Pkg.update()
    ENV["PYTHON"]=""; Pkg.build("PyCall")
    ] activate .; instantiate
    ```

4. (Optional) Install packages globally from julia REPL:
    ```julia
    add LightGraphs BlossomV ChpSim OrderedCollections PyPlot
    ```

## Usage

Run the following Julia code on (slurm) server:

```bash
# On command line
# For interative testing:
# julia --project=. -e 'import Pkg; Pkg.instantiate(); ENV["SLURM_ARRAY_TASK_ID"]=1; ENV["SLURM_CPUS_PER_TASK"]=1; include("src/run.jl")'

# Running bash script:
sbatch jobs_slurm.sh
```


Run the following Julia code in a REPL (start one with `julia --project=.`)

```julia
#Setup

include("make_plots_distributed.jl")

using .MakePlots

println("Ready to set up workers.")
flush(stdout)
MakePlots.setup() # setup workers based on cpu counts

#Run simulation

d_tasks = 1 # 1 - 3

job_id = "rz_0.001"
samples = 1000 * 30000 # 20,000 samples take about 1min per cpu (1 plot, 1 dist, 10 angles)
dists = [(1 + 2 * d_tasks):2:(2 + 2 * d_tasks)...] # code distance: 3 or 5 or 7 separately
plots = [4] # list of plot types
plot_id = 4 # type of plots to be made

println("Simulation begins... (", samples, " samples)")
flush(stdout)

results = MakePlots.dist_calc_all(job_id, dists, samples, plots) # start computing and wait until complete
println("Simulation completed.")

collected = MakePlots.fetch_finished(results, job_id, plot_id)
println(collected)

```

