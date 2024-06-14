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

struct MyLogPotential
    n_matches::Int
    n_players::Int
    winner_ids::Vector{Int}
    loser_ids::Vector{Int}
end

function (log_potential::MyLogPotential)(params::Vector)
    player_sd = abs(params[log_potential.n_players + 1])
    lp = 0.0
    log_likelihood = 0.0
    for n in 1:log_potential.n_matches
        log_likelihood += log(StatsFuns.logistic(params[log_potential.winner_ids[n]]*player_sd - 
        params[log_potential.loser_ids[n]]*player_sd))
    end
    
    for i in 1:log_potential.n_players
        lp -= 0.5*(params[i]*player_sd)^2
    end
    
    return (log_likelihood + player_sd*lp)
end
function Pigeons.initialization(log_potential::MyLogPotential, rng::AbstractRNG, _::Int64) 
    return randn(rng, log_potential.n_players + 1)
end
function Pigeons.sample_iid!(log_potential::MyLogPotential, replica, shared)
    randn!(replica.rng, replica.state)
end

const stan_data = [
    29,
    8,
    [1,1,1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,3,4,4,4,4,5,5,5,6,6,7,8],
    [2,3,4,5,6,7,8,3,4,5,6,7,8,4,5,6,7,2,5,6,2,1,6,7,8,2,8,1,2]]
    
const n_matches = stan_data[1]
const n_players = stan_data[2]
const winner_ids = stan_data[3]
const loser_ids = stan_data[4]

LogDensityProblems.dimension(lp::MyLogPotential) = n_players + 1
LogDensityProblems.logdensity(lp::MyLogPotential,x) = lp(x)

function main()
    #include("./data/fetch_data.jl")

    #data = create_arrays()
    #winner_ids = get(data,"winner_ids",nothing)
    #loser_ids = get(data,"loser_ids",nothing)
    #player_encoder = get(data,"player_encoder",nothing)

    #stan_data = [
    #length(winner_ids),
    #length(player_encoder.classes_),
    #winner_ids .+ 1,
    #loser_ids .+ 1]
    

    log_potential = MyLogPotential(n_matches, n_players, winner_ids, loser_ids)
    pt = @time pigeons(target=log_potential, reference = MyLogPotential(0,8,[1,1,1,1,1,1,1,1],[2,2,2,2,2,2,2,2]),
    record=[traces;record_default()])#, explorer=AutoMALA())
    #report(pt)
end
