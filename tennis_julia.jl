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

#stan_file = "./model/model.stan"



#json_data = JSON.json(stan_data)

#print("pigeons start")

invlogit(z::Real) = 1/(1+exp(-z))

struct MyLogPotential
    n_matches::Int
    n_players::Int
    winner_ids::Vector{Int}
    loser_ids::Vector{Int}
end

function (log_potential::MyLogPotential)(params)
    player_sd = abs(params[log_potential.n_players + 1])
    #player_skills_raw = @view params[1:log_potential.n_players]
    log_likelihood = 0
    if(player_sd < 0)
        player_sd = 0
    end
    for n in 1:log_potential.n_matches

        log_likelihood += log(invlogit(params[log_potential.winner_ids[n]]*player_sd - 
        params[log_potential.loser_ids[n]]*player_sd))
    end
    lp = 0
    for i in 1:log_potential.n_players
        lp -= 0.5*(params[i]*player_sd)^2
    end
    return (log_likelihood + player_sd*lp)
end
function Pigeons.initialization(log_potential::MyLogPotential, rng::AbstractRNG, _::Int64) 
    params = randn(rng, log_potential.n_players+1)
    return params
end
function Pigeons.sample_iid!(log_potential::MyLogPotential, replica, shared)
    randn!(replica.rng, replica.state)
end

stan_data = [
    17,
    4,
    [1,2,1,1,3,1,4,4,2,2,3,2,4,4,3,3,4],
    [2,1,2,3,1,4,1,1,3,3,2,4,2,2,4,4,3]]

    n_matches = stan_data[1]
    n_players = stan_data[2]
    winner_ids = stan_data[3]
    loser_ids = stan_data[4]

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
    pt = @time pigeons(target=log_potential, reference = MyLogPotential(0,4,[1,1,1,1],[2,2,2,2]),
    record=[traces;record_default()])#, explorer=AutoMALA())
    #report(pt)
end
