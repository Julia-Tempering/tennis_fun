include("sackmann.jl")
using CSV
x = get_data("./tennis_atp")
CSV.write("get_data.csv",x)
