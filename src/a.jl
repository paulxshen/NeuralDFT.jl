using DFTK
using LinearAlgebra
using Printf
using FileIO
using UnPack
using PeriodicTable
using DataStructures
# ]add DFTK,DataStructures,PeriodicTable

function compute_scfres(atoms, lattice; Ecut=40, ψ=nothing, ρ=nothing, kgrid=[1, 1, 1], tol=1e-8)
    model = model_LDA(lattice, atoms)
    basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid)
    if ρ === nothing
        ρ = ρv_SAD = guess_density(basis)
    end
    scfres = self_consistent_field(basis; ψ=ψ, ρ=ρ,
        tol)

    @unpack ρ = scfres
    ρv = ρ
    ρv[:, :, :, 1], ρv_SAD[:, :, :, 1]

end

function read_xyz(ifile::String)
    """
    Reads in an xyz file of possibly multiple geometries, returning the header, atom labels,
    and coordinates as arrays of strings and Float64s for the coordinates.
    """
    @time file_contents = readlines(ifile)
    header = Array{String,1}()
    atom_labels = Array{Array{String,1},1}()
    geoms = Array{Array{Float64,2},1}()
    for (i, line) in enumerate(file_contents)
        if isa(tryparse(Int, line), Int)
            # allocate the geometry for this frame
            N = parse(Int, line)
            head = string(N)
            labels = String[]
            # store the header for this frame
            head = string(line, file_contents[i+1])
            i += 1
            push!(header, head)
            # loop through the geometry storing the vectors and atom labels as you go
            geom = zeros((3, N))
            for j = 1:N
                coords = split(file_contents[i+1])
                i += 1
                push!(labels, coords[1])
                geom[:, j] = parse.(Float64, coords[2:4])
            end
            push!(geoms, geom)
            push!(atom_labels, labels)
        end
    end
    return header, atom_labels, geoms
end

psp = Dict([
    1 => ElementPsp(:H, psp=load_psp("hgh/lda/h-q1")),
    2 => ElementPsp(:He, psp=load_psp("hgh/lda/He-q2")),
    3 => ElementPsp(:Li, psp=load_psp("hgh/lda/Li-q3")),
    6 => ElementPsp(:C, psp=load_psp("hgh/lda/c-q4")),
    7 => ElementPsp(:N, psp=load_psp("hgh/lda/n-q5")),
    8 => ElementPsp(:O, psp=load_psp("hgh/lda/o-q6")),
    9 => ElementPsp(:F, psp=load_psp("hgh/lda/f-q7")),
    10 => ElementPsp(:Ne, psp=load_psp("hgh/lda/Ne-q8")),
    14 => ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4")),
    16 => ElementPsp(:S, psp=load_psp("hgh/lda/s-q6")),
    17 => ElementPsp(:Cl, psp=load_psp("hgh/lda/Cl-q7")),
])
He = psp[2]
Ne = psp[10]
Si = psp[14]

name = :Si
cases = []

a = 8.0
lattice = a * [1 0 0; 0 1 0; 0 0 1.0]
lattice = a * Diagonal(ones(3))

rvecs = [ones(3) / 2,]
rvecs = [lattice \ [0., 0., 0.], lattice \ [1.4, 0., 0.]]
positions = hcat([lattice * v for v in rvecs]...)
atoms = [psp[1] => rvecs]

model = model_LDA(lattice, atoms)
kgrid = [4,4,4]     # k-point grid (Regular Monkhorst-Pack grid)
Ecut = 40              # kinetic energy cutoff
basis = PlaneWaveBasis(model; Ecut, kgrid)

# 3. Run the SCF procedure to obtain the ground state
scfres = self_consistent_field(basis, tol=1e-8)
global ρv = scfres.ρ[:, :, :, 1]

# cell = hcat((eachcol(lattice) ./ size(ρv))...)
# Z = [14, 14]
# Zc = [10, 10]
# Zv = Z - Zc
# periodic = false
# case = (; cell, ρ, ρv, ρv_SAD, ρc_SAD, Z, Zc, Zv, positions, lattice, Ecut, kgrid, periodic)
# push!(cases, case)

# save("..\\data\\$name.jld2", "cases", cases)
volume(ρv)