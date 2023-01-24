module Jobs

using Distributed
using OrderedCollections

export launch_workers, 
    current_results, current_times, current_time_info, run_on_workers


capacity = 10000
const jobs = RemoteChannel(()->Channel{Tuple{Any, Function, Any}}(capacity))
const results = RemoteChannel(()->Channel{Pair{Tuple{Any, Any}, Tuple}}(
                    capacity))
const collected = OrderedDict()


function kill_workers(; waitfor=30)
    ws = workers()
    (length(ws) == 1 && 1 in ws) && return
    rmprocs(ws, waitfor=waitfor)
end

function clear_jobs()
    while isready(jobs)
        take!(jobs)
    end
    nothing
end

function take_results()
    while isready(results)
        (job_id, job_args), (
            result, t, bytes, gctime, memallocs) = take!(results)
        collected[job_id][job_args] = (result, t, bytes, gctime, memallocs)
    end
    nothing
end

function launch_workers(n::Integer)
    num_new = n - length(setdiff!(Set(workers()), [1]))
    # Create worker processes
    new_workers = addprocs(num_new; exeflags="--project")
    length(workers()) == n || error("didn't start the right number of workers")
    1 in workers() && error("main process is listed as a worker")
    nothing
end

function current_results(job_id)
    take_results()
    OrderedDict(
         k => val
         for (k, (val,)) in collected[job_id]
    )
end

function current_times(job_id)
    take_results()
    OrderedDict(
         k => time
         for (k, (val, time)) in collected[job_id]
    )
end

function current_time_info(job_id)
    take_results()
    OrderedDict(
         k => v[2:end]
         for (k, v) in collected[job_id]
    )
end

function fetch_results_internal(job_id, plot_i, default::Tuple{Float64,Float64,Float64,Float64,Float64,Float64,Float64}=(1.0,1.0,1.0,1.0,1.0,1.0,1.0))
    x_arr = x_axis_for_plot(plot_i)
    y_arr_dict = Dict{Int, Vector{Tuple{Float64,Float64,Float64,Float64,Float64,Float64,Float64}}}()
    results = current_results(job_id)
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

function run_on_workers(job_id, job_f, job_args_list)
    job_id in keys(collected) && error("job_id <$job_id> has already been used")

    1 in workers() && error("run launch_workers(n) first")

    # Set up workerpool
    wp = CachingPool(workers())

    # Ensure do_work is defined
    @everywhere function _do_work(job_id, job_f, job_args, results)
        print("Worker $(myid()) begins: ", job_args, "\n")
        flush(stdout)
        val, t, bytes, gctime, memallocs = @timed job_f(job_args...)
        put!(results,
             (job_id, job_args) => (val, t, bytes, gctime, memallocs))
        
        print("Worker $(myid()) is done: ", job_args, "\n")
        flush(stdout)
    end

    # Start worker tasks
    @sync for job_args in job_args_list
        @async remotecall_wait(Main._do_work, wp, job_id, job_f, job_args, results)
    end

    # Wait and Collect results
    time_taken = reduce((x,y)-> x+y, 
             let tt = v
               tt
             end
             for (_,v) in current_times(job_id) 
    )
    println(time_taken, " seconds")
    current_results(job_id)

end


end  # module
