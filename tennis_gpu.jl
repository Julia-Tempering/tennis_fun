using BridgeStan
using Pigeons
using CSV
using DataFrames
using JSON
using InferenceReport
using Distributions
using Random
using StatsFuns
using LogDensityProblems
using BenchmarkTools
using Profile
using PProf
using ProfileCanvas
using Enzyme
using CUDA

CUDA.allowscalar(false)  

global WINNER_IDS = Ref{CuArray{Int, 1}}()
global LOSER_IDS = Ref{CuArray{Int, 1}}()


function invlogit(x::AbstractVector)
    return 1 ./ (1 .+ exp.( .- x))
end
struct MyLogPotential
    n_matches::Int
    n_players::Int
    winner_ids::CuArray{Int, 1}
    loser_ids::CuArray{Int,1}
end

function (log_potential::MyLogPotential)(params::Vector)
    copyto!(PARAMS, Float32.(params))
    player_sd = abs(params[log_potential.n_players + 1])
    lps = ((params .* player_sd) .^ 2) .* 0.5
    lp = -sum(lps)
    log_likelihoods = log.(invlogit(PARAMS[log_potential.winner_ids] .* player_sd - PARAMS[log_potential.loser_ids] .* player_sd))
    log_likelihood = CUDA.reduce(+,log_likelihoods)
    return (log_likelihood + player_sd*lp)
end
function Pigeons.initialization(log_potential::MyLogPotential, rng::AbstractRNG, _::Int64) 
    return randn(rng, log_potential.n_players + 1)
end
function Pigeons.sample_iid!(log_potential::MyLogPotential, replica, shared)
    randn!(replica.rng, replica.state)
end

n_matches = 29
n_players = 8
global PARAMS = CuArray{Float32, 1}(undef, n_players+1)

#include("./fetch_data.jl")

#const data = create_arrays()
#const winner_ids = get(data,"winner_ids",nothing) .+ 1
#const loser_ids = get(data,"loser_ids",nothing) .+ 1
#const player_encoder = get(data,"player_encoder",nothing)
#const n_players = length(player_encoder.classes_)
#const n_matches = length(winner_ids)

    #stan_data = [
    #length(winner_ids),
    #length(player_encoder.classes_),
    #winner_ids .+ 1,
    #loser_ids .+ 1]

LogDensityProblems.dimension(lp::MyLogPotential) = n_players + 1
LogDensityProblems.logdensity(lp::MyLogPotential,x) = lp(x)


function main()

    WINNER_IDS[] = CuArray(Int.(Vector([1,1,1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,3,4,4,4,4,5,5,5,6,6,7,8])))
    LOSER_IDS[] = CuArray(Int.(Vector([2,3,4,5,6,7,8,3,4,5,6,7,8,4,5,6,7,2,5,6,2,1,6,7,8,2,8,1,2])))

    log_potential = MyLogPotential(n_matches, n_players, WINNER_IDS[], LOSER_IDS[])
    pt = @time pigeons(target=log_potential, reference = MyLogPotential(0,8,[1,1,1,1,1,1,1,1],[2,2,2,2,2,2,2,2]),
    record=[traces;record_default()])#, explorer=AutoMALA(default_autodiff_backend = :Enzyme))
    #report(pt)
end
