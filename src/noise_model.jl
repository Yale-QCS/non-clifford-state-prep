export make_noise_model_for_paper


### Make the noise model used in the paper
# Function "" can overwrite the noise parameters.
p0 = 0.6 # can be overwritten
starting_model = Dict{Symbol, Float64}(
    :t1_t => 100_000,  # ns
    :t1_c => 1_000_000,  # ns
    :dur_t => 50,  # ns
    :dur_tt => 200,  # ns
    :dur_tc => 200,  # ns
    :dur_loadstore => 150,  # ns
    :dur_meas => 200,  # ns
    :p_t => p0, # single-qubit error rate
    :p_tt => p0, # two-qubit error rate, could be a factor of 10 wrt p_t
    :p_tc => p0,
    :p_loadstore => p0,
    :p_meas => p0, # measurement error rate
    :cavity_depth => 10,
)
const_keys = [:dur_tt, :dur_t, :dur_tc, :dur_loadstore, :dur_meas,
              :cavity_depth]
coherence_keys = [:t1_t, :t1_c]
error_rate_keys = [:p_t, :p_tt, :p_tc, :p_loadstore, :p_meas]
sensitivity_base_p = 2e-3

function make_noise_model_for_paper(base_error::Float64, override_pairs=())
    error_factor = base_error / p0
    actual_model = Dict{Symbol, Float64}()
    for k in const_keys
        actual_model[k] = starting_model[k]
    end
    for k in coherence_keys
        actual_model[k] = starting_model[k] / error_factor
    end
    for k in error_rate_keys
        actual_model[k] = starting_model[k] * error_factor
    end
    for (k, v) in override_pairs
        actual_model[k] = v
    end
    NoiseModel(actual_model)
end


### Util functions
"""
    combine_flip_probs(p1, p2, ...)

Return the probability that an odd number of events occur
with the given probabilities.
"""
function combine_flip_probs(p_flip::Float64...)
    p_total = 0
    for p in p_flip
        p_total = p_total + p - 2*p_total*p
    end
    p_total
end

"""
    combine_error_probs(p1, p2, ...)

Return the probability that at least one event occurs
with the given probabilities.
"""
function combine_error_probs(p_err::Float64...)
    1 - prod(1 .- p_err)
end

function coherence_error(t1, duration)
    1 - MathConstants.e ^ (-duration / t1)
end


### Calculate edge weights and simulation noise parameters

function calculate_qubit_error_single_pauli(model::NoiseModel;
        t_t::Float64 = 0.0,
        t_c::Float64 = 0.0,
        n_t::Int = 0,
        n_tt::Int = 0,
        n_tc::Int = 0,
        n_loadstore::Int = 0,
        n_meas::Int = 0,
    )
    p = combine_flip_probs(
        2/3 * coherence_error(model.t1_t, t_t),
        2/3 * coherence_error(model.t1_c, t_c),
        repeat([2/3*model.p_t], n_t)...,
        repeat([8/15*model.p_tt], n_tt)...,
        repeat([8/15*model.p_tc], n_tc)...,
        repeat([2/3*model.p_loadstore], n_loadstore)...,
        repeat([model.p_meas], n_meas)...,
    )
    p
end

