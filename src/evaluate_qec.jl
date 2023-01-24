using Random
using LightGraphs
using BlossomV

using ChpSim
include("BiMaps.jl"); using .BiMaps

export x_gate!, z_gate!, reset!, measure_reset!, rz_gate!,
    NoiseModel, SyndromeCircuit, BasicSyndrome,
    LogicalOpSim, LogicalOpRun,
    MatchingGraphWeights, apply_sim_error!, do_single_logical_op_run, do_n_logical_op_runs


const rng = Random.GLOBAL_RNG

function Base.append!(dict::Dict, iter)
    for (k, v) in iter
        dict[k] = v
    end
end


# Extra methods
function x_gate!(state::ChpState, qubit::Int)
    hadamard!(state, qubit)
    phase!(state, qubit)
    phase!(state, qubit)
    hadamard!(state, qubit)
end
function z_gate!(state::ChpState, qubit::Int)
    phase!(state, qubit)
    phase!(state, qubit)
end
function reset!(state::ChpState, qubit::Int)
    if measure!(state, qubit, bias=0).value
        x_gate!(state, qubit)
    end
    nothing
end
function measure_reset!(state::ChpState, qubit::Int)
    meas = measure!(state, qubit)
    if meas.value
        x_gate!(state, qubit)
    end
    meas
end



# New non-cliford gate, returns a list of sampled bits of length dist.
# rz is translated as a stochastic Z flips along the logical operators
function rz_gate!(state::ChpState, theta::Float64, dist::Int, z_qubits::Vector{Int})
    r = rand(rng, Float64, dist)
    pI = cos(theta/2)^2
    pZ = sin(theta/2)^2
    projected=BitArray(undef, dist)
    for i in 1:dist
        if r[i] < pI
            projected[i]=0
        end 
        if r[i] >= pI
            projected[i]=1
            z_gate!(state, z_qubits[i])
        end
    end
    pI, pZ, projected
end


"""
    NoiseModel(...)

Holds probabilities of various quantum error sources
"""
struct NoiseModel
    errors::Dict{Symbol, Float64}
end
function Base.getproperty(noise::NoiseModel, s::Symbol)
    s == :errors && return getfield(noise, :errors)
    getfield(noise, :errors)[s]
end

"""
    SyndromeCircuit

Parent type representing a style of syndrome measurement circuit.
"""
abstract type SyndromeCircuit end
"""
    BasicSyndrome

Basic syndrome measurement with simple errors
"""
struct BasicSyndrome <: SyndromeCircuit end


NodeT = Tuple{Int, Symbol, Int, Int}
PlaqInfoT = NamedTuple{(:ancilla, :data), Tuple{Int, Vector{Int}}}
#####################

struct LogicalOpSim
    z_dist::Int
    x_dist::Int
    m_dist::Int
    theta::Float64
    syndrome_circuit::SyndromeCircuit
    noise_model::NoiseModel
    num_qubits::Int
    anc_qubits::Vector{Int}
    z_anc_qubits::Vector{Int}
    x_anc_qubits::Vector{Int}
    data_qubits::Vector{Int}
    logical_z_qubits::Vector{Int}
    diagonal_x_ancilla::Vector{Int}
    #logical_op_projection::AbstractArray{Bool}
    z_plaqs::Vector{Tuple{Int, Int}}
    z_plaq_info::Vector{PlaqInfoT}
    z_space_boundary::Vector{Tuple{Int, Int}}
    z_doubled_boundary::Vector{Tuple{Int, Int}}
    z_graph_nodes::BiMap{NodeT, Int}
    z_graph::Graph{Int}
    z_costs::Matrix{Float64}
    z_bpaths::Set{Tuple{Int, Int}}
    z_path_lengths::Dict{Tuple{Int, Int}, Int}
    x_plaqs::Vector{Tuple{Int, Int}}
    x_plaq_info::Vector{PlaqInfoT}
    x_space_boundary::Vector{Tuple{Int, Int}}
    x_doubled_boundary::Vector{Tuple{Int, Int}}
    x_graph_nodes::BiMap{NodeT, Int}
    x_graph::Graph{Int}
    x_costs::Matrix{Float64}
    x_bpaths::Set{Tuple{Int, Int}}
    x_path_lengths::Dict{Tuple{Int, Int}, Int}
end

function LogicalOpSim(dist::Int, theta::Float64, syndrome_circuit::SyndromeCircuit,
                         noise_model::NoiseModel)
    LogicalOpSim(dist, dist, dist, theta, syndrome_circuit, noise_model)
end
function LogicalOpSim(z_dist::Int, x_dist::Int, m_dist::Int, theta::Float64,
                         syndrome_circuit::SyndromeCircuit, noise_model::NoiseModel)
    # Graphs
    (z_plaqs, z_space_boundary, z_doubled_boundary), (
        x_plaqs, x_space_boundary, x_doubled_boundary) = (
        make_plaqs(z_dist, x_dist))
    # Qubits
    num_qubits, anc_qubits, data_qubits, z_plaq_info, x_plaq_info, lz_qubits, z_anc_qubits, x_anc_qubits = (
        make_qubit_assignments(z_dist, z_plaqs, x_dist, x_plaqs))
    z_graph_nodes, z_graph = (
        make_graph(m_dist+1, z_plaqs, z_space_boundary, false, false))
    x_graph_nodes, x_graph = (
        make_graph(m_dist+1, x_plaqs, x_space_boundary, false, false))

    z_costs, z_bpaths, z_path_lengths = constuct_graph_costs(
        z_graph_nodes, z_graph, z_doubled_boundary,
        noise_model, syndrome_circuit, false, z_dist, m_dist)
    x_costs, x_bpaths, x_path_lengths = constuct_graph_costs(
        x_graph_nodes, x_graph, x_doubled_boundary,
        noise_model, syndrome_circuit, true, x_dist, m_dist)
    #lz_proj = BitArray(undef, z_dist) # dummy value for now
    
    diagonal_anc = diagonal_x_plaqs(x_plaq_info, lz_qubits)

    LogicalOpSim(
        z_dist, x_dist, m_dist, theta, syndrome_circuit, noise_model,
        num_qubits, anc_qubits, z_anc_qubits, x_anc_qubits,
        data_qubits, lz_qubits, diagonal_anc,
        z_plaqs, z_plaq_info, z_space_boundary, z_doubled_boundary,
            z_graph_nodes, z_graph, z_costs, z_bpaths, z_path_lengths,
        x_plaqs, x_plaq_info, x_space_boundary, x_doubled_boundary,
            x_graph_nodes, x_graph, x_costs, x_bpaths, x_path_lengths,
    )
end



#####################

function make_plaqs(z_dist, x_dist)
    z_plaqs = Tuple{Int, Int}[
        (x, y)
        for x in 1:z_dist-1
        for y in 0:x_dist
        if mod(x+y, 2) == 0
    ]
    z_space_boundary = Tuple{Int, Int}[
        (x, y)
        for (x, y) in z_plaqs
        if x == 1 || x == z_dist-1
    ]
    z_doubled_boundary = Tuple{Int, Int}[  # Plaqs with two boundary qubits
        (x, y)
        for (x, y) in z_space_boundary
        if 1 <= y <= x_dist-1
    ]
    x_plaqs = Tuple{Int, Int}[
        (x, y)
        for x in 0:z_dist
        for y in 1:x_dist-1
        if mod(x+y, 2) == 1
    ]
    x_space_boundary = Tuple{Int, Int}[
        (x, y)
        for (x, y) in x_plaqs
        if y == 1 || y == x_dist-1
    ]
    x_doubled_boundary = Tuple{Int, Int}[  # Plaqs with two boundary qubits
        (x, y)
        for (x, y) in x_space_boundary
        if 1 <= x <= z_dist-1
    ]
    (z_plaqs, z_space_boundary, z_doubled_boundary), (
        x_plaqs, x_space_boundary, x_doubled_boundary)
end
function make_qubit_assignments(z_dist, z_plaqs, x_dist, x_plaqs)
    counter = Iterators.Stateful(Iterators.countfrom(1))
    data_qubits = Dict{Tuple{Int, Int}, Int}(
        (x, y) => popfirst!(counter)
        for x in 1:z_dist
        for y in 1:x_dist
    )
    lz_qubits = Dict{Tuple{Int, Int}, Int}(
        (z_dist-lz+1, lz) => data_qubits[(z_dist-lz+1, lz)]
        for lz in 1:z_dist
    )
    anc_qubits = Dict{Tuple{Int, Int}, Int}()
    append!(anc_qubits, xy => popfirst!(counter) for xy in z_plaqs)
    z_anc_qubits = [anc_qubits[xy] for xy in z_plaqs]
    append!(anc_qubits, xy => popfirst!(counter) for xy in x_plaqs)
    x_anc_qubits = [anc_qubits[xy] for xy in x_plaqs]
    num_qubits = length(data_qubits) + length(anc_qubits)
    make_plaq_info(plaqs)::Vector{PlaqInfoT} = [
        begin
            qubits = Int[
                data_qubits[(xx, yy)]
                for (xx, yy) in [(x, y), (x, y+1), (x+1, y), (x+1, y+1)]
                if (xx, yy) in keys(data_qubits)
            ]
            @assert(length(qubits) in (2, 4), "Invalid plaquettes")
            (ancilla=anc_qubits[(x, y)], data=qubits)
        end
        for (x, y) in plaqs
    ]
    z_plaq_info = make_plaq_info(z_plaqs)
    x_plaq_info = make_plaq_info(x_plaqs)
    (num_qubits, collect(values(anc_qubits)), collect(values(data_qubits)),
        z_plaq_info, x_plaq_info, collect(values(lz_qubits)), z_anc_qubits, x_anc_qubits)
end
function make_graph(m_dist, plaqs, space_boundary,
                    start_boundary::Bool, end_boundary::Bool)
    graph_nodes = BiMap{NodeT, Int}()
    counter = Iterators.Stateful(Iterators.countfrom(1))
    # Assign node ids
    if start_boundary
        for (x, y) in plaqs
            graph_nodes[(0, :tboundary, x, y)] = popfirst!(counter)
        end
    end
    for t in 1:m_dist
        for (x, y) in plaqs
            graph_nodes[(t, :plaq, x, y)] = popfirst!(counter)
        end
        for (x, y) in space_boundary
            graph_nodes[(t, :sboundary, x, y)] = popfirst!(counter)
        end
    end
    if end_boundary
        for (x, y) in plaqs
            graph_nodes[(m_dist+1, :tboundary, x, y)] = popfirst!(counter)
        end
    end
    # Make graph
    plaq_set = Set(plaqs)
    graph = Graph{Int}(length(graph_nodes))
    # Space edges
    for t in 1:m_dist
        for (x, y) in plaqs
            for (x2, y2) in [(x-1, y-1), (x-1, y+1), (x+1, y-1), (x+1, y+1)]
                (x2, y2) in plaq_set || continue
                add_edge!(graph,
                          graph_nodes[(t, :plaq, x, y)],
                          graph_nodes[(t, :plaq, x2, y2)])
            end
        end
        for (x, y) in space_boundary
            add_edge!(graph,
                      graph_nodes[(t, :plaq, x, y)],
                      graph_nodes[(t, :sboundary, x, y)])
            # Connect boundaries
            if (x, y) != space_boundary[1]
                add_edge!(graph,
                          graph_nodes[(t, :sboundary, x, y)],
                          graph_nodes[(t, :sboundary, space_boundary[1]...)])
            end
        end
    end
    # Time edges
    for t in 1-start_boundary:m_dist-1+end_boundary
        type1 = t == 0 ? :tboundary : :plaq
        type2 = t == m_dist ? :tboundary : :plaq
        for (x, y) in plaqs
            add_edge!(graph,
                      graph_nodes[(t, type1, x, y)],
                      graph_nodes[(t+1, type2, x, y)])
        end
        # Connect boundaries
        if 0 < t < m_dist
            add_edge!(graph,
                      graph_nodes[(t, :sboundary, space_boundary[1]...)],
                      graph_nodes[(t+1, :sboundary, space_boundary[1]...)])
        end
    end
    if start_boundary
        # Connect boundaries
        for (x, y) in plaqs
            add_edge!(graph,
                      graph_nodes[(0, :tboundary, x, y)],
                      graph_nodes[(1, :sboundary, space_boundary[1]...)])
        end
    end
    if end_boundary
        # Connect boundaries
        for (x, y) in plaqs
            add_edge!(graph,
                      graph_nodes[(m_dist+1, :tboundary, x, y)],
                      graph_nodes[(m_dist, :sboundary, space_boundary[1]...)])
        end
    end
    graph_nodes, graph
end
function diagonal_x_plaqs(x_plaq_info, logical_z_qubits)
    # check if neighbor has lz_qubits
    plaqs = x_plaq_info
    lz_qubits = logical_z_qubits
    res = Vector{Int}()
    for p in plaqs
        #println(p[:data])
        if size(p[:data]) == (4,)
            for d in p[:data]
                if d in lz_qubits
                    append!(res, p[:ancilla])
                    break
                end
            end
        end
    end
    res
end

"""
Used by construct_graph_costs().
"""
struct MatchingGraphWeights{SyndromeCircuitT} <: AbstractMatrix{Float64}
    graph_nodes::BiMap{NodeT, Int}
    doubled_boundary::Set{Tuple{Int, Int}}
    noise_model::NoiseModel
    syndrome_circuit::SyndromeCircuitT
    space_edge1::Float64
    time_edge1::Float64
    space_edge::Float64
    time_edge::Float64
end
Base.size(w::MatchingGraphWeights) = (l=length(w.graph_nodes); (l, l))
function Base.getindex(w::MatchingGraphWeights, i, j)::Float64
    r = rev(w.graph_nodes)
    n, m = r[i], r[j]
    # Inter-boundary
    (n[2] != :plaq && m[2] != :plaq) && return 0.0
    # Time edge (including boundary)
    n[1] != m[1] && return (min(n[1], m[1]) == 0 ? w.time_edge1 : w.time_edge)
    # Space edge (including boundary)
    return (n[1] == 1 ? w.space_edge1 : w.space_edge)
    #n[2] != :plaq && ((n, m) = (m, n))  # Make it plaq->boundary or plaq->plaq
    #return (m[2] != :plaq && (m[3], m[4]) in w.doubled_boundary
    #    # Plaq with two boundary qubits
    #    ? w.space_double_edge
    #    # Edge error probability dependent on only one qubit
    #    : w.space_edge
    #)
end
function matching_space_edge(::SyndromeCircuit, noise::NoiseModel,
                             is_x::Bool, m_dist::Int, fisrt_layer::Bool)
    -log(noise.errors[:uniform_data])
end
function matching_time_edge(::SyndromeCircuit, noise::NoiseModel,
                            is_x::Bool, m_dist::Int, first_layer::Bool)
    -log(noise.errors[:uniform_anc])
end

function constuct_graph_costs(graph_nodes::BiMap{NodeT, Int}, graph::Graph{Int},
                              doubled_boundary::Vector{Tuple{Int, Int}},
                              noise_model::NoiseModel,
                              syndrome_circuit::SyndromeCircuit,
                              is_x::Bool, dist::Int, m_dist::Int)
    rev_graph_nodes = rev(graph_nodes)
    weights = MatchingGraphWeights(
        graph_nodes, Set(doubled_boundary), noise_model, syndrome_circuit,
        matching_space_edge(syndrome_circuit, noise_model, is_x, m_dist, true),
        matching_time_edge(syndrome_circuit, noise_model, is_x, m_dist, true),
        matching_space_edge(syndrome_circuit, noise_model, is_x, m_dist, false),
        matching_time_edge(syndrome_circuit, noise_model, is_x, m_dist, false),
    )
    paths = floyd_warshall_shortest_paths(graph, weights)
    boundary_ids = Set{Int}(
        id
        for ((_, kind, _, _), id) in graph_nodes
        if kind != :plaq
    )
    boundary_paths = Set{Tuple{Int, Int}}()
    path_parities = Dict{Tuple{Int, Int}, Int}()
    for i in 1:length(graph_nodes)-1
        for j in i+1:length(graph_nodes)
            #i >= j && continue
            if i in boundary_ids && j in boundary_ids
                push!(boundary_paths, (i, j))
                push!(boundary_paths, (j, i))
                continue
            end
            jj = j
            non_boundary_count = 0
            hits_boundary = false
            while jj != 0
                in_boundary = jj in boundary_ids
                jj2 = paths.parents[i, jj]
                is_time_edge = (
                    !in_boundary && jj2 != 0 && !(jj2 in boundary_ids)
                    && rev_graph_nodes[jj][1] != rev_graph_nodes[jj2][1]
                )
                non_boundary_count += !in_boundary && !is_time_edge
                hits_boundary |= in_boundary
                jj = jj2
            end
            if hits_boundary
                push!(boundary_paths, (i, j))
                push!(boundary_paths, (j, i))
            end
            path_parity = non_boundary_count - !hits_boundary
            if !(i in boundary_ids && j in boundary_ids)
                path_parities[(i, j)] = path_parities[(j, i)] = path_parity
            end
        end
    end
    paths.dists, boundary_paths, path_parities
end


struct LogicalOpRun
    ctx::LogicalOpSim
    state::ChpState
    zx_error_counts::Vector{Int}
    zx_meas_error_counts::Vector{Int}
    error_indices_one::Vector{Int}
    error_indices_two::Vector{Int}
    z_prev::Vector{Bool}
    x_prev::Vector{Bool}
    z_syndromes::Matrix{Bool}
    x_syndromes::Matrix{Bool}
    sim_noise_params::Dict{Symbol, Float64}
end
function LogicalOpRun(ctx::LogicalOpSim)
    state = ChpState(ctx.num_qubits, bitpack=false)
    zx_error_counts = Int[0, 0]
    zx_meas_error_counts = Int[0, 0]
    error_indices_one = Int[0, 0]
    error_indices_two = Int[0, 0]
    z_prev = zeros(Bool, length(ctx.z_plaqs))
    x_prev = zeros(Bool, length(ctx.x_plaqs))
    z_syndromes = Matrix{Bool}(undef, length(z_prev), 2) # error-free once and error detect twice
    x_syndromes = Matrix{Bool}(undef, length(x_prev), 2) # error-free once and error detect twice
    sim_noise_params = simulation_noise_parameters(ctx.syndrome_circuit,
                                                   ctx.noise_model, ctx)
    LogicalOpRun(
        ctx, state, zx_error_counts, zx_meas_error_counts, error_indices_one, error_indices_two,
        z_prev, x_prev, z_syndromes, x_syndromes,
        sim_noise_params,
    )
end



function simulation_noise_parameters(::SyndromeCircuit, model::Nothing,
                                     ctx::LogicalOpSim)
    nothing
end


function simulation_noise_parameters(::SyndromeCircuit, model::NoiseModel,
                                     ctx::LogicalOpSim)
    Dict{Symbol, Float64}(
        p_data_layer1 => 0,
        p_data => model.p_t,
        p_anc_z => model.p_t,
        p_anc_x => model.p_t,
        p_cnot1 => 0,
        p_cnot => 0,
    )
end


function apply_sim_error!(noise_params::Nothing,
                          state::ChpState, zx_counts_out::Vector{Int},
                          kind::Symbol, qubits::NTuple{N, Int} where N)
    # Apply no error when no noise model
    return false
end
function apply_sim_error!(noise_params::Dict{Symbol, Float64},
                          state::ChpState, zx_counts_out::Vector{Int},
                          kind::Symbol, qubits::NTuple{N, Int} where N)
    p = noise_params[kind]
    z_count = x_count = 0
    kstring = string(kind)
    #println("kind: ", kstring, " p: ", p)
    for q in qubits
        r = rand(rng, Float64)
        r >= p && return false
        # Apply X, Y, or Z with (1/3)p probability
        if r < (2/3)*p
            z_gate!(state, q)
            z_count += 1
        end
        if r >= (1/3)*p
            x_gate!(state, q)
            x_count += 1
        end
    end
    zx_counts_out[2] += z_count  # Z errors cause X syndromes
    zx_counts_out[1] += x_count
    #println(x_count > 0 || z_count > 0)
    return x_count > 0 || z_count > 0
end


function exec_syndrome_layer!(
        noise_model::Union{NoiseModel, Nothing},
        syndrome_circuit::SyndromeCircuit,
        run::LogicalOpRun,
        layer_i::Int)
    ctx = run.ctx
    state = run.state
    # Apply errors
    noise_params = simulation_noise_parameters(syndrome_circuit, noise_model, ctx)
    #p_data = layer_i == 1 ? :p_data_layer1 : :p_data
    p_data = :p_data
    for q in ctx.data_qubits
        err_flag = apply_sim_error!(noise_params, state, run.zx_error_counts,
                          p_data, (q,))
        if err_flag
            if layer_i == 1
                append!(run.error_indices_one, q)
            else
                append!(run.error_indices_two, q)
            end
        end
    end
    # Run circuit
    for info in ctx.z_plaq_info
        anc = info.ancilla
        for dat in info.data
            cnot!(state, dat, anc)
        end
    end
    for info in ctx.x_plaq_info
        anc = info.ancilla
        hadamard!(state, anc)
        for dat in info.data
            cnot!(state, anc, dat)
        end
        hadamard!(state, anc)
    end
    
    # Inject noise on ancilla after a cycle of x/z-plaq error detection complete
    for info in ctx.z_plaq_info
        #apply_sim_error!(noise_params, state, run.zx_meas_error_counts,
        #                  :p_anc_z, (info.ancilla,))
        err_flag = apply_sim_error!(noise_params, state, run.zx_meas_error_counts,
                          p_data, (info.ancilla,))
        if err_flag 
            #println("x: ", x_cnt, "z: ", z_cnt, "z_plaq: ", info)
            if layer_i == 1
                append!(run.error_indices_one, info.ancilla)
            else
                append!(run.error_indices_two, info.ancilla)
            end
        end
    end
    for info in ctx.x_plaq_info
        err_flag = apply_sim_error!(noise_params, state, run.zx_meas_error_counts,
                          p_data, (info.ancilla,))
        if err_flag 
            #println("x: ", x_cnt, "z: ", z_cnt, "x_plaq: ", info)
            if layer_i == 1
                append!(run.error_indices_one, info.ancilla)
            else
                append!(run.error_indices_two, info.ancilla)
            end
        end
    end
    # Measure
    for (i, info) in enumerate(ctx.z_plaq_info)
        meas = measure_reset!(state, info.ancilla).value  # TODO: rng
        if layer_i == 1
            run.z_syndromes[i, layer_i] = run.z_prev[i] ⊻ meas
        else
            run.z_syndromes[i, layer_i] = run.z_prev[i] ⊻ meas
        end
        run.z_prev[i] = meas
    end
    for (i, info) in enumerate(ctx.x_plaq_info)
        meas = measure_reset!(state, info.ancilla).value  # TODO: rng
        if layer_i == 1
            run.x_syndromes[i, layer_i] = run.x_prev[i] ⊻ meas
        else
            run.x_syndromes[i, layer_i] = run.x_prev[i] ⊻ meas
        end
        run.x_prev[i] = meas
    end
    nothing
end


function reset_all_qubits!(state::ChpState)
    state.x .= 0
    state.x[CartesianIndex.(1:state.n, 1:state.n)] .= 1  # Primary diagonal
    state.z .= 0
    # Diagonal offset down by n
    state.z[CartesianIndex.(state.n+1:2state.n, 1:state.n)] .= 1
    state.r .= 0
    state
end

function reset_all!(run::LogicalOpRun)
    reset_all_qubits!(run.state)
    run.zx_error_counts .= 0
    run.zx_meas_error_counts .= 0
    filter!(e -> e<0, run.error_indices_one)
    filter!(e -> e<0, run.error_indices_two)
end

"""
    simulate_logical_op_syndrome_run(run)

Simulate some (e.g., m_dist) rounds of syndrome measurement (with start and end boundaries).
"""

function simulate_logical_op_syndrome_run(run::LogicalOpRun)
    simulate_logical_op_syndrome_run(run.ctx.syndrome_circuit, run)
end
function simulate_logical_op_syndrome_run(syndrome_circuit::SyndromeCircuit, run::LogicalOpRun)
    ctx = run.ctx
    reset_all!(run)  # Clean start
    state = run.state
    run.zx_error_counts .= 0
    run.zx_meas_error_counts .= 0
    filter!(e -> e<0, run.error_indices_one) # clear all
    filter!(e -> e<0, run.error_indices_two) 
    # Logical operation
    for q in ctx.data_qubits
        hadamard!(state, q) # transversal hadamard
    end
    # Noise free first layer
    exec_syndrome_layer!(nothing, syndrome_circuit, run, 1)
    #exec_syndrome_layer!(nothing, syndrome_circuit, run, 2)

    _, _, projected = rz_gate!(state, ctx.theta, ctx.z_dist, ctx.logical_z_qubits)
    # Error detect twice (with noise)
    for i in 1:2
        # Inject noise and stabilizer measure twice 
        noise = ctx.noise_model
        exec_syndrome_layer!(noise, syndrome_circuit, run, i)
    end
    op_zx_error_counts = copy(run.zx_error_counts)
    ms_zx_error_counts = copy(run.zx_meas_error_counts)
    run.z_syndromes, run.x_syndromes, projected, op_zx_error_counts, ms_zx_error_counts 
end


function syndromes_to_error_ids(syndromes, plaqs, graph_nodes)
    @assert size(syndromes)[1] == length(plaqs)
    Int[
        graph_nodes[(t, :plaq, plaqs[i]...)]
        for t in axes(syndromes, 2)
        for i in axes(syndromes, 1)
        if syndromes[i, t]
    ]
end

function syndromes_to_all_errors(syndromes, anc_ids)
    @assert size(syndromes)[1] == length(anc_ids) # same order as plaqs
    Int[
        anc_ids[i]
        for t in axes(syndromes, 2)
        for i in axes(syndromes, 1)
        if syndromes[i, t]
    ]
end

function construct_matching_graph(graph_nodes, costs, extra_boundary::NodeT,
                                  error_ids::Vector{Int})
    node_count = div(length(error_ids) + 1, 2) * 2  # Round up to even
    edge_count = div(node_count * (node_count-1), 2)
    matching = Matching(Float64, node_count, edge_count)
    if node_count > length(error_ids)  # Round up to even
        push!(error_ids, graph_nodes[extra_boundary])
    end
    for i in 1:node_count-1#length(error_locations)-1
        for j in i+1:node_count#length(error_ids)
            c = costs[error_ids[i], error_ids[j]]
            add_edge(matching, i-1, j-1, c)
        end
    end
    matching
end

function count_corrected_errors(matching, error_ids, path_lengths)
    count = 0
    for i in 1:length(error_ids)-1
        j = get_match(matching, i-1) + 1  # get_match is zero-indexed
        i < j || continue  # Count each pair once
        count += path_lengths[error_ids[i], error_ids[j]]
    end
    count
end

function match_and_evaluate_syndromes(plaqs, graph_nodes, space_boundary,
                                      costs, path_lengths,
                                      syndromes, error_count)
    error_ids = syndromes_to_error_ids(syndromes, plaqs, graph_nodes)
    extra_boundary = (1, :sboundary, space_boundary[1]...)  # Doesn't matter which
    matching = construct_matching_graph(graph_nodes, costs, extra_boundary, error_ids)
    solve(matching)
    corrected = count_corrected_errors(matching, error_ids, path_lengths)
    finalize(matching)
    failed = mod(error_count + corrected, 2) != 0
    failed
end

#############################################


function do_single_logical_op_run(run::LogicalOpRun, z_only::Bool=false)
    z_syndromes, x_syndromes, projected, op_zx_error_counts, ms_zx_error_counts  = simulate_logical_op_syndrome_run(run)
    # if syndromes along the diagonal region, discard: todo
    discarded = false
    logical_err = false
    #z_errors, x_errors = run.zx_error_counts
    
    ctx = run.ctx
    x_ancs = ctx.x_anc_qubits
    z_ancs = ctx.z_anc_qubits
    x_syn = copy(x_syndromes)
    z_syn = copy(z_syndromes)
    z_error_ids = syndromes_to_all_errors(x_syndromes, x_ancs)
    x_error_ids = syndromes_to_all_errors(z_syndromes, z_ancs)

    if length(z_error_ids) > 0 || length(x_error_ids) > 0 # discard if any error detected
        discarded = true
    end
    if discarded == false
        theta = ctx.theta
        amp_I = cos(theta/2)
        amp_Z = sin(theta/2)
        dist = ctx.z_dist
        weight = sum(projected)
        if 0 < weight < dist
            logical_err = true
            #println(projected)
        end
	# taking min and max to count coherently I^d and Z^d correctly with high probability
        a = min(weight, dist-weight)
        b = max(weight, dist-weight)
        # Actual logical space Z expected value sin(theta_L/2)
        actual_op = (amp_I^(a)*amp_Z^(b)) / (sqrt(amp_I^(2*b)*amp_Z^(2*a) + amp_Z^(2*b)*amp_I^(2*a))) # updated: sin^(d-w)cos^w in numerator 
        # Actual angle is theta_L
        actual_angle = 2 * asin(actual_op) # allowed domain for sin is -pi/2 to pi/2
        actual_i = im^(b-a) # an im factor for every amp_Z in numerator
    else
        actual_angle = 0.0
        actual_i = im
    end

    discarded, logical_err, actual_angle, actual_i, z_syn, x_syn, projected, op_zx_error_counts, ms_zx_error_counts 
end

function do_n_logical_op_runs(run::LogicalOpRun, n::Int, z_only::Bool=false)
    fail_count = 0
    save_count = 0
    logical_err_count = 0
    ctx = run.ctx
    diagonal_anc = ctx.diagonal_x_ancilla
    theta = ctx.theta
    amp_I = cos(theta/2)
    amp_Z = sin(theta/2)
    dist = ctx.z_dist
    # Predicted logical space Z expected value for sin(theta_L/2)
    ideal_op = (amp_Z^dist)/(sqrt(amp_I^(2*dist) + amp_Z^(2*dist)))
    # To account for i factor, denominator of ideal_op gets i*-i in amp_Z, ideal_i is the numerator factor
    ideal_i = im^dist # for odd dist, we get +/-im
    # Predict angle is theta_L
    ideal_angle = 2 * asin(ideal_op) # allowed domain for sin is -pi/2 to pi/2
    actual_avg = 0.0
    weighted_err = 0.0
    # For calculating running variance with Welford online algorithm
    mtwo = 0.0 # for weight err
    amtwo = 0.0 # for actual average angle
    for _ in 1:n
        discarded, logical_err, actual_angle, actual_i, z_syndromes, x_syndromes, projected, op_zx_error_counts, ms_zx_error_counts  = do_single_logical_op_run(run)
        if ideal_i != actual_i
            actual_angle = 0.0 - actual_angle
        end
        if discarded
            fail_count += 1
        elseif logical_err
            save_count += 1
            logical_err_count += 1
            println(ideal_angle, ", ", actual_angle)
            println(run.ctx.logical_z_qubits)
            println(projected)
            println("Data error counts (x, z): ", op_zx_error_counts)
            println("Meas error counts (x, z): ", ms_zx_error_counts)
            println("Errors 1: ", run.error_indices_one)
            println("Errors 2: ", run.error_indices_two)
            actual_err = (sin((ideal_angle-actual_angle) / 2))^2
            delta = actual_err - weighted_err
            weighted_err += delta / save_count
            deltatwo = actual_err - weighted_err
            mtwo += delta * deltatwo

            adelta = actual_angle - actual_avg
            actual_avg += adelta / save_count
            adeltatwo = actual_angle - actual_avg
            amtwo += adelta * adeltatwo

        else
            save_count += 1
            adelta = actual_angle - actual_avg
            actual_avg += adelta / save_count
            adeltatwo = actual_angle - actual_avg
            amtwo += adelta * adeltatwo
        end
    end
    logical_err_rate = n == fail_count ? 0.0 : logical_err_count / (n-fail_count)
    fail_rate = fail_count / n
    avg_angle = actual_avg
    angle_std = save_count == 0 ? 0.0 : amtwo / save_count
    angle_std = sqrt(angle_std)
    logical_infidelity = weighted_err
    infidelity_std = save_count == 0 ? 0.0 : mtwo / save_count
    infideltiy_std = sqrt(infidelity_std)
    return fail_rate, ideal_angle, avg_angle, angle_std, logical_infidelity, infidelity_std, logical_err_rate 
end        


