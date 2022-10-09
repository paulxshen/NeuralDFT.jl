using FileIO
using GLMakie
using Random
using StatsBase
using PeriodicTable
Random.seed!(1)

include("density.jl")

# dir="..\\density-prediction\\data"
name="qm9_4"
name="Si"
data = load("data\\$name.jld2", "cases")

# p=train(data)
# save("model_$name.jld2","p",p)
for t in data[1:1]
    global   ρc_SAD,lattice,Z,positions,Ecut,ρv,periodic
    @unpack  ρc_SAD,lattice,Z,positions,Ecut,ρv,periodic=t
    # @unpack  ρc_SAD,lattice,Z,positions,Ecut,ρv,=t
    # periodic=false
     ρv_=predict_density(Z,positions,lattice;Ecut,periodic)
    #  ρv_=predict_density(Z,positions,lattice;Ecut,p)
     @show nae(ρv_,ρv)

    
    println()
end
# volume(ρc_SAD)