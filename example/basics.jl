"""
we predict density and forces on equilibrium and distorted geometries of CH4
"""

using FileIO
using Random
using Statistics
using UnPack
Random.seed!(1)
using GLMakie:volume
using LinearAlgebra
using EquivariantOperators
include("../src/pretrained.jl")

# include("../src/NeuralDFT.jl")
# using .NeuralDFT
# include("../../EquivariantOperators.jl/src/EquivariantOperators.jl")

# whether to re-instantiate models or load them from cache file
reset=true
# reset=false

# CH4 atomic numbers
Z = [6, 1, 1, 1, 1]

# positions in Bohr (1 Angstrom = 1.88973 Bohr)
pos0 = 1.88973 * [
    -0.0126981359 1.0858041578 0.0080009958
    0.002150416 -0.0060313176 0.0019761204
    1.0117308433 1.4637511618 0.0002765748
    -0.540815069 1.4475266138 -0.8766437152
    -0.5238136345 1.4379326443 0.9063972942]'
center!(pos0)
# pretrained predictor of small organic molecules

if !reset
    d = "C:\\Users\\xingpins"
   global  predictor = load("$d\\demo2_predictor.jld2", "predictor")
else
    resolution = 0.15
    cell = resolution * I(3)
    d="."
    # model = load("$d\\demo2.jld2", "model")
     model = DEMO2
    global predictor = DensityPredictor(cell, model,)
    d = "C:\\Users\\xingpins"
    save("$d\\demo2_predictor.jld2", "predictor", predictor)
end

# 8 x 8 x 8 Bohr box domain
domain = 8 * I(3)
origin = :center
periodic = false
mode = :verbose

# density prediction result
pos=pos0
res = predictor(Z, pos, domain; periodic, origin, mode)
@unpack ρ, grid = res
s=volume(ρ)
display(s)
# save("ch4.png",s)

# forces in Hartrees / Bohr
d = "C:\\Users\\xingpins"
if !reset
    calc = load("$d\\calc.jld2", "calc",)
else
    cell=.1I(3)
    calc=ForceField(cell)
    save("$d\\calc.jld2","calc",calc)
end

forces = calc(Z,pos,ρ;origin,cell=predictor.cell,)
@info "forces near equilibrium"
@info "CH length: $(norm(pos[:, 1] - pos[:, 2])) Bohr"
display(forces)

# compress geometry and recalculate forces
pos = 0.9pos0
@info "forces for compressed geometry"
@info "CH length: $(norm(pos[:, 1] - pos[:, 2])) Bohr"
ρ= predictor(Z, pos, domain; periodic, origin)
forces = calc(Z,pos,ρ;origin,cell=predictor.cell,)
display(forces)

s=volume(ρ)
display(s)
# save("ch4c.png",s)

# geometry relaxation loop - needs to fix stability issues
# for i = 1:12
#     ρ = predictor(Z, pos, domain; periodic, origin)
#     forces .= calc(Z, pos, ρ; origin, cell=predictor.cell)
#     pos .+= .2forces
#     display(forces)
#     @show norm(pos[:, 1] - pos[:, 2])
# end
