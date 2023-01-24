"""
#Setup
include("src/make_plots_distributed.jl"); using .MakePlots
MakePlots.setup() # setup workers based on cpu counts

#Run simulation
job_id = "rz_0.001"
samples = 100*cores*100
dists = [3:2:7] # code distance
plots = [4] # list of plot_id (type of plots)
plot_id = 4
results = MakePlots.dist_calc_all(job_id, dists, samples, plots) # start computing and wait until complete
MakePlots.fetch_finished(results, job_id, plot_id)

"""
module MakePlots

using Distributed
# import PyPlot; plt = PyPlot

export setup, dist_calc_all, fetch_finished, plot, plot_finished

num_cores = parse(Int, ENV["SLURM_CPUS_PER_TASK"])
#num_cores = 8

include("sim.jl")
include("jobs_sync.jl"); using .Jobs

function setup(; num_workers=num_cores)
    println("Launching $(num_workers) workers")
    Jobs.launch_workers(num_workers)
    eval(:(@everywhere (
        include("sim.jl");
        using .SIM.BiMaps
    )))

    @everywhere function everywhere_calc_single(
            plot_i, syndrome_sym, d, e, th, override_key, samples)
        step = samples
        e *= 9 <= plot_i <=11 ? 1.0e9 : 1.0
        if override_key === nothing
            override_pairs = ()
        else
            override_pairs = [(override_key, e)]
            e = SIM.sensitivity_base_p
        end

        # Shortcut for slow high error runs
        (e > 0.01 && d > 5) && (samples = min(samples, 10000))
        e > 0.03 && (samples = min(samples, 1000))
        e >= 0.1 && (samples = min(samples, 500))

        model = SIM.make_noise_model_for_paper(e, override_pairs)

        ctx = SIM.LogicalOpSim(d, th,
            getproperty(SIM, syndrome_sym)(),
            model
        )
        run = SIM.LogicalOpRun(ctx)
        loops = div(samples, step) # error bar calculated, so loops is one. 
        reduce((x, y) -> x .+ y, 
                let (r, s, t, tbar, lf, lfbar, le) = SIM.do_n_logical_op_runs(run, step, false)
                    ccall(:jl_gc_collect, Nothing, ())
                    (r, s, t, tbar, lf, lfbar, le)
                end
                for _ in 1:loops
        ) ./ loops
    end
end


function x_axis_for_plot(plot_i)
    if plot_i == 1
        10 .^ LinRange(log10(0.1), log10(0.0001), 19)[4:end-2]
    elseif 2 <= plot_i <= 3
        2 .^ LinRange(log2(1/2), log2(1/1048576), 20)
    elseif plot_i == 4
        [x for x in LinRange(pi/16,pi/8,10)]
    end
end
function confs_for_plot(plot_i, dists, samples)
    if plot_i == 1
        confs_for_thresh_plot(plot_i, dists, samples)
    elseif 2 <= plot_i <= 3
        confs_for_logical_op_plot(plot_i, dists, samples)
    elseif plot_i == 4
        confs_for_cost_plot(plot_i, dists, samples)
    end
end
function confs_for_thresh_plot(plot_i, dists, samples)
    x_arr = x_axis_for_plot(plot_i)
    syndrom_sym = :SimpleSyndrome
    th = pi/8
    [
        (plot_i, syndrom_sym, d, e, th, nothing, samples)
        for e in x_arr
        for d in dists
    ]
end
function confs_for_logical_op_plot(plot_i, dists, samples)
    x_arr = x_axis_for_plot(plot_i)
    e = 0.006
    [
        (plot_i, :SimpleSyndrome, d, e, th, nothing, samples)
        for th in x_arr
        for d in dists
    ]
end
function confs_for_cost_plot(plot_i, dists, samples)
    x_arr = x_axis_for_plot(plot_i)
    e = 0.001
    [
        (plot_i, :SimpleSyndrome, d, e, th, nothing, samples)
        for th in x_arr
        for d in dists
    ]
end
function dist_calc_all(job_id, dists, samples, which_plots=1:12)
    confs = []
    for i in which_plots
        append!(confs, confs_for_plot(i, dists, samples))
    end
    Jobs.run_on_workers(job_id, Main.everywhere_calc_single, confs)
end

function fetch_finished(results, job_id, plot_i, default::Tuple{Float64,Float64,Float64,Float64,Float64,Float64,Float64}=(1.0,1.0,1.0,1.0,1.0,1.0,1.0))
    x_arr = x_axis_for_plot(plot_i)
    y_arr_dict = Dict{Int, Vector{Tuple{Float64,Float64,Float64,Float64,Float64,Float64,Float64}}}()
    # results = Jobs.current_results(job_id)
    for ((res_plot_i, _, d, e, th,  _, _), val) in results
        res_plot_i == plot_i || continue
        if !(d in keys(y_arr_dict))
            y_arr_dict[d] = repeat([default], length(x_arr))
        end
        if plot_i == 1
            i = indexin([e], x_arr)[1]
        elseif 2 <= plot_i <= 3
            i = indexin([th], x_arr)[1]
        elseif plot_i == 4
            i = indexin([th], x_arr)[1]
        end
        i === nothing || (y_arr_dict[d][i] = val)
    end
    dists = sort!(collect(keys(y_arr_dict)))
    y_arrs = [y_arr_dict[d] for d in dists]
    x_arr, dists, y_arrs
end

function plot(plot_i, x_arr, dists, y_arrs)
     nothing
end


function plot_finished(job_id, plot_i, default::Float64=1.0)
    x_arr, dists, y_arrs = fetch_finished(job_id, plot_i, default)
    plot(plot_i, x_arr, dists, y_arrs)
end



end
