using FileIO
# using GLMakie
using Random
using StatsBase
using Statistics
using PeriodicTable
# using Plots
Random.seed!(1)

include("properties.jl")

# dir="..\\density-prediction\\data"

name="qm9_1101a"
name="qm9_1029"
# name="Si"
data = load("..\\data\\$name.jld2", "cases")

# @unpack domain, origin, ρev = data[1]
# volume(ρev)

predictor=train(data[:],nsamples=150)
# histogram(p)
# summarystats(p)
d="."
# d="C:\\Users\\xingpins"
save("$d\\demo1.jld2","model",predictor.model)
# save("model_$name.jld2","p",p)

 t =data[1]
# for t in data[1:1]
@unpack domain,Z,pos,ρev,periodic,origin=t
sz=size(ρev)
ρres=predict_density(Z,pos,domain,sz;verbose=true,periodic,origin,model=MODEL)
@unpack ρev,ρzv,X,grid=ρres
forces=predict_forces(ρres)
@show forces
@show nae(ρres.ρev,t.ρev)

# sum(X[1])*ρres.grid.dv
# display(volume(.1X[1]))
# display(volume(X[10]))
# display(volume(X[15]))
# display(volume(ρzv))
# display(volume(ρres.ρev))
# display(volume(t.ρev))
