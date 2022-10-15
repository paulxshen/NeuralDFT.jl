using FileIO
using GLMakie
using Random
using StatsBase
using Statistics
using PeriodicTable
using Plots
Random.seed!(1)

include("density.jl")

# dir="..\\density-prediction\\data"
name="qm9_4"
# name="Si"
data = load("data\\$name.jld2", "cases")

p=train(data)
histogram(p)
summarystats(p)
# save("model_$name.jld2","p",p)
# for t in data[1:1]
#     global   ρc_SAD,lattice,Z,positions,resolution,ρv,periodic
#     @unpack  ρc_SAD,lattice,Z,positions,resolution,ρv,periodic=t
#     # @unpack  ρc_SAD,lattice,Z,positions,resolution,ρv,=t
#     # periodic=false
#      ρv_=predict_density(Z,positions,lattice;resolution,periodic)
#     #  ρv_=predict_density(Z,positions,lattice;resolution,p)
#      @show nae(ρv_,ρv)

    
#     println()
# end
# volume(ρc_SAD)