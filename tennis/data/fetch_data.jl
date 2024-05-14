using DataFrames
using CSV
using ScikitLearn
@sk_import preprocessing: LabelEncoder
encoder = LabelEncoder()
include("sackmann.jl")

function create_arrays(start_year::Int=1960,
                        data_dir::String="./tennis_atp",
                        inc_qual_chal::Bool=false,
                        inc_futures::Bool=false)

    df = get_data(data_dir)
    rel_df = df[year.(df.tourney_date) .>= start_year, :]
    ScikitLearn.fit!(encoder, vcat(rel_df.winner_name,rel_df.loser_name))

    winner_ids = ScikitLearn.transform(encoder,rel_df.winner_name)
    loser_ids = ScikitLearn.transform(encoder,rel_df.loser_name)

    return Dict(
        "winner_ids" => winner_ids,
        "loser_ids" => loser_ids,
        "player_encoder" => encoder
    )
end

