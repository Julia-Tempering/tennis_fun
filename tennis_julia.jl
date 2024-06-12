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

invlogit(z::Real) = 1/(1+exp(-clamp(z,-709,709)))

struct MyLogPotential
    n_matches::Int
    n_players::Int
    winner_ids::Vector{Int}
    loser_ids::Vector{Int}
end

function (log_potential::MyLogPotential)(params)
    player_sd = abs(params[log_potential.n_players + 1])
    player_skills_raw = params[1:log_potential.n_players]
    #player_skills_raw, player_sd = params
    log_likelihood = 0
    if(player_sd < 0)
        player_sd = 0
    end

    player_skills = player_skills_raw * player_sd
    for n in 1:log_potential.n_matches
        #dist = Distributions.BernoulliLogit(player_skills[log_potential.winner_ids[n]] - 
        #player_skills[log_potential.loser_ids[n]])

        pred = invlogit(player_skills[log_potential.winner_ids[n]] - 
            player_skills[log_potential.loser_ids[n]])
        log_likelihood += log(pred)
    end
    lp = 0
    for i in 1:log_potential.n_players
        lp -= 0.5*player_skills[i]^2
    end
    return (log_likelihood + player_sd*lp)
end
function Pigeons.initialization(log_potential::MyLogPotential, rng::AbstractRNG, _::Int64) 
    params = randn(rng, log_potential.n_players+1)
    return params
end
function Pigeons.sample_iid!(log_potential::MyLogPotential, replica, shared)
    rng = replica.rng
    #state = @view replica.state[1:log_potential.n_players]
    randn!(rng, replica.state)
    #replica.state = new_state
end





#pt = pigeons(target = StanLogPotential(stan_file,json_data), 
#   record = [traces;round_trip;record_default()], 
 #   explorer = SliceSampler(w=5))


#report(pt)

LogDensityProblems.dimension(lp::MyLogPotential) = 5
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

    stan_data = [
    17,
    4,
    [1,2,1,1,3,1,4,4,2,2,3,2,4,4,3,3,4],
    [2,1,2,3,1,4,1,1,3,3,2,4,2,2,4,4,3]]

    n_matches = stan_data[1]
    n_players = stan_data[2]
    winner_ids = stan_data[3]
    loser_ids = stan_data[4]


    log_potential = MyLogPotential(n_matches, n_players, winner_ids, loser_ids)
    pt = @time pigeons(target=log_potential, reference = MyLogPotential(0,4,[1,1,1,1],[2,2,2,2]),
    record=[traces;record_default()], explorer=AutoMALA())
    report(pt)
end
