using CSV
using DataFrames
using Glob
using StringEncodings
using Dates

# read file with specified encoding and return a DataFrame
function read_csv_with_encoding(file_path, encoding="ISO-8859-1")
    file_content = open(read, file_path)  # read file as raw bytes
    decoded_content = decode(file_content, encoding)  # decode using "ISO-8859-1"
    return CSV.File(IOBuffer(decoded_content), types=Dict(:winner_hand => Union{String, Missing})) |> DataFrame
end

function get_data(
        sackmann_dir;
        tour="atp",
        keep_davis_cup=false,
        discard_retirements=true,
        include_qualifying_and_challengers=false,
        include_futures=false
    )
    all_csvs = glob("*$(tour)_matches_????.csv", sackmann_dir)

    if include_qualifying_and_challengers
        append!(all_csvs, glob("*$(tour)_matches_qual_chall_????.csv", sackmann_dir))
    end

    if include_futures
        append!(all_csvs, glob("*$(tour)_matches_futures_????.csv", sackmann_dir))
    end

    sort!(all_csvs, by=x -> parse(Int, splitext(basename(x))[1][end-3:end]))

    levels_to_drop = String[]
    if !include_futures
        push!(levels_to_drop, "S")
    end
    if !include_qualifying_and_challengers
        push!(levels_to_drop, "C")
    end
    if !keep_davis_cup
        push!(levels_to_drop, "D")
    end

    data = DataFrame()

    for file in all_csvs
        df = read_csv_with_encoding(file, "ISO-8859-1")
        allowmissing!(df, :winner_hand)
        dropmissing!(df, [:winner_name, :loser_name, :score])

        if discard_retirements
            filter!(row -> !occursin(r"RET|W/O|DEF|nbsp|Def.", row.score), df)
        end

        filter!(row -> length(row.score) > 4, df)
        filter!(row -> !in(row.tourney_level, levels_to_drop), df)

        append!(data, df, promote=true)  
    end

    round_numbers = Dict(
        "R128" => 1,
        "RR" => 1,
        "R64" => 2,
        "R32" => 3,
        "R16" => 4,
        "QF" => 5,
        "SF" => 6,
        "F" => 7
    )

    filter!(row -> haskey(round_numbers, row.round), data)
    transform!(data, :round => ByRow(round -> round_numbers[round]) => :round_number)

    transform!(data, :tourney_date => ByRow(tourney_date -> Date(string(tourney_date), "yyyymmdd")) => :tourney_date)
    transform!(data, :tourney_date => ByRow(year) => :year)

    sort!(data, [:tourney_date, :round_number])

    data.pts_won_serve_winner = data.w_1stWon .+ data.w_2ndWon
    data.pts_won_serve_loser = data.l_1stWon .+ data.l_2ndWon
    data.pts_played_serve_winner = data.w_svpt
    data.pts_played_serve_loser = data.l_svpt
    data.spw_winner = (data.w_1stWon .+ data.w_2ndWon) ./ data.w_svpt
    data.spw_loser = (data.l_1stWon .+ data.l_2ndWon) ./ data.l_svpt
    data.spw_margin = data.spw_winner .- data.spw_loser

    return data
end
