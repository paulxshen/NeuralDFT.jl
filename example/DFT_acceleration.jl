"""
we compare electron density predicted by ML vs computed by DFT, noting reduction in SCF iterations by starting with an ML density vs SAD density
"""

using DFTK
using LinearAlgebra
using FileIO
using UnPack
using PeriodicTable
using DataStructures
using EquivariantOperators
# ]add DFTK,DataStructures,PeriodicTable

# include("../src/NeuralDFT.jl")
# using .NeuralDFT
include("../src/pretrained.jl")
include("utils_DFTK.jl")

# whether to re-instantiate models or load them from cache file
reset = true
reset=false

# atomic numbers
# Z = [6, 6, 8, 1, 1, 1, 1]
Z = [6, 6, 8, 1, 1, 1, 1, 1, 1]

# positions in Bohr (1 Angstrom = 1.88973 Bohr)
# pos = 1.88973 * [
#     -0.0029448212 1.5099136648 0.0086727849;
#     0.0260828384 0.0032756259 -0.037459115;
#     0.9422880119 -0.6550703513 -0.4568257611;
#     0.9227880213 1.926342418 -0.3914655687;
#     -0.8620154031 1.878524808 -0.5647953841;
#     -0.1505063787 1.8439338318 1.0428910048;
#     -0.8944300885 -0.4864340773 0.3577486492;
# ]'
pos = 1.88973 * [
    -0.0086050396 1.5020382883 -0.0068121748;
    0.0109931006 -0.0176487687 -0.013770355;
    0.6808884095 -0.4404180314 -1.1931320987;
    1.0115333417 1.896620303 -0.0192015475;
    -0.5315986204 1.8807610923 -0.8897465868;
    -0.5167457158 1.876117708 0.8871073864;
    0.5237712083 -0.3891229547 0.8882408183;
    -1.0202754406 -0.4050726108 0.0169067033;
    0.6952957263 -1.4017956761 -1.2014849452;
]'

# slightly offset the positions so origin is near the molecule's center
center!(pos)

# a x a x a lattice
a = 12
lattice = a * I(3)

# offset the positions so origin is at (1, 1, 1) of lattice array with the molecule centered in lattice
pos .+= a / 2

##==========
# DFT calculation with SAD initial density

# DFT parameters, refer to DFTK.jl docs
Ecut = 40
kgrid = [1, 1, 1]
tol = 1e-2

atoms = DefaultOrderedDict(Vector)
for (a, x) in zip(Z, eachcol(pos))
    xr = lattice \ x
    push!(atoms[psp[a]], xr)
end
atoms = collect(atoms)

model = model_LDA(lattice, atoms)
basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid)
ρ = ρ_SAD = guess_density(basis)
scfres = self_consistent_field(basis; ρ,
    tol)
# ρ_SAD=guess_density(basis)
forces = compute_forces_cart(scfres)
forces = hcat(vcat(forces...)...)

@unpack energies, ρ = scfres
ρ_DFT = ρ[:, :, :, 1]
ρ_SAD = ρ_SAD[:, :, :, 1]

s=volume(ρ_DFT)
# display(s)
save("etoh.png",s)

s=volume(ρ_SAD)
# display(s)
save("etohsad.png",s)
##==========================

##===============
# repeat DFT with ML predicted density

d = "C:\\Users\\xingpins"
if !reset #
    predictor = load("$d\\demo1_predictor.jld2", "predictor")
else
    resolution = 0.15
    cell = resolution * I(3)
    model = DEMO1
    predictor = DensityPredictor(cell, model)
    save("$d\\demo1_predictor.jld2", "predictor", predictor)
end

domain = lattice
sz = size(ρ_SAD)
origin = ones(3)
periodic = false

# density prediction result
ρ_ML = predictor(Z, pos, domain, sz; periodic, origin)
@show nae(ρ_ML, ρ_DFT)
@show nae(ρ_SAD, ρ_DFT)

s=volume(ρ_ML)
# display(s)
save("etohml.png",s)

# rerun DFT
# ρ = reshape(ρ_ML, sz..., 1)
# scfres = self_consistent_field(basis; ρ,
#     tol);