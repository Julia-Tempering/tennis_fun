using BridgeStan
using Pigeons
using CSV
using DataFrames
using JSON
using InferenceReport

include("fetch_data.jl")

data = create_arrays()
winner_ids = get(data,"winner_ids",nothing)
loser_ids = get(data,"loser_ids",nothing)
player_encoder = get(data,"player_encoder",nothing)

stan_file = "./model/model.stan"

stan_data = Dict(
    "n_matches" => length(winner_ids),
    "n_players" => length(player_encoder.classes_),
    "winner_ids" => winner_ids .+ 1,
    "loser_ids" => loser_ids .+ 1
)

json_data = JSON.json(stan_data)

pt = pigeons(target = StanLogPotential(stan_file,json_data), 
record = [traces;round_trip;record_default()])

report(pt)