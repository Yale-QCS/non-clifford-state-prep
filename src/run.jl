#Setup

include("make_plots_distributed.jl")

using .MakePlots

println("Ready to set up workers.")
flush(stdout)
MakePlots.setup() # setup workers based on cpu counts

#Run simulation

d_tasks = parse(Int, ENV["SLURM_ARRAY_TASK_ID"]) # 1 - 3

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


