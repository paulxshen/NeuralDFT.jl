using UnPack
using FileIO
using LinearAlgebra
# using GLMakie
using Random
using StatsBase
using DataStructures
using PeriodicTable
using DFTK
# using EquivariantOperators
include("../../EquivariantOperators.jl/src/operators.jl")
include("utils.jl")
Random.seed!(1)

const p = load("model.jld2", "p")

function rescale(a, s)
    a / sum(a) * s
end

function mix!(X)
    n = length(X)
    for i = 1:n
        for j = i:n
            push!(X, X[i] .* X[j])
        end
    end
end

# function predict_density(atoms,positions,lattice,resolution,periodic)
function features(Z, positions, lattice, resolution, periodic)
    sz=norm.(eachcol(lattice)).÷resolution

    Zv = broadcast(x ->
            elements[x].shells[end],Z)
    Zc = Z - Zv
    zpos = DefaultDict(Vector)
    zposc = DefaultDict(Vector)
    ρ_SAD = zeros(sz)
    ρc_SAD = zeros(sz)

    for (z,zc,p) in zip(Z,Zc,eachcol(positions))
        push!( zpos[z],p)
        push!( zposc[zc],p)
    end
    for (z,zc) in zip(unique(Z),unique(Zc))
        ρ=zeros(sz)
        for p in zpos[z]
            put!(ρ,p,1)
        end
        # ρ_SAD+=ρops[z](ρ)
        σ=atom_decay_length(zv,zc)
        ρ_SAD+=z*Gaussian(cell,σ,3σ;pad,border)(ρ)
       
        ρ=zeros(sz)
        for p in zposc[zc]
            put!(ρ,p,1)
        end
        ρc_SAD+=ρops[zc](ρ)
        # σ=elements[zc].atomic_radius
        σ=atom_decay_length(zc,core(zc))
        ρc_SAD+=zc*Gaussian(cell,σ,3σ;pad,border)(ρ)
    end

    core_atoms = DefaultDict(Vector)
    for (a, b, x) in zip(Z, Zc, eachcol(positions))
        xr = lattice \ x
        push!(atoms[psp[a]], xr)
        if b != 0
            push!(core_atoms[psp[b]], xr)
        end
    end
    atoms = collect(atoms)
    core_atoms = collect(core_atoms)

    model = model_LDA(lattice, atoms)
    basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid)
    ρv_SAD = guess_density(basis)

    if !isempty(core_atoms)
        core_model = model_LDA(lattice, core_atoms)
        core_basis = PlaneWaveBasis(core_model; Ecut, kgrid)
        ρc_SAD = guess_density(core_basis)
    else
    end

    ρc_SAD = ρc_SAD[:, :, :, 1]
    ρv_SAD = ρv_SAD[:, :, :, 1]

    sz = size(ρv_SAD)
    cell = lattice ./ sz
    origin = ones(3)
    # origin =periodic ? ones(3) : sz./2
    grid = Grid(cell, sz; origin)
    @unpack dv = grid

    N = sum(Z)
    Nv = sum(Zv)
    Nc = sum(Zc)
    @assert N == Nv + Nc
    @show sum(ρv_SAD) * dv, Nv
    @show sum(ρc_SAD) * dv, Nc

    ρpv = zeros(sz)
    put!(ρpv, grid, positions, Zv)
    ρpc = zeros(sz)
    put!(ρpc, grid, positions, Zc)
    ρp = ρpc + ρpv

    # ρmh = zeros(sz)
    # put!(ρmh, grid, positions, [ustrip(elements[z].molar_heat) for z in Z])
    ρen = zeros(sz)
    put!(ρen, grid, positions, [EN[z] for z in Z])

    # rmax = 3.0
    pad = :same
    border = periodic ? :circular : 0
    alg = :fft
    s = [0.5, 1]
    # s = [0.5, 1, 1.5]
    # s = [1, 2]
    ops = vcat(
        [Op(r -> exp(-r / a), 6a, cell; pad, border, alg) for a = s],
        [Op(r -> exp(-(r / a)^2), 3a, cell; pad, border, alg) for a = s],
        [Op(r -> 1 / r, 6.0, cell; pad, border, alg, rmin=1e-6)],
    )

    X0 = [vec([f(u) for f = ops, u = (ρp, ρpv, ρen, ρc_SAD, ρv_SAD)])..., ρc_SAD, ρv_SAD]
    # X0 = [vec([f(u) for f = ops, u = (ρp,  ρpv, ρen)])..., ρc_SAD,ρv_SAD]
    # X0_ = copy(X0)
    mix!(X0)

    # mix!(X)
    # X = [X0..., [op(ρv) for op in ops[[]]]...]
    X0
end

function predict_density(atoms, positions, lattice; Ecut=40, periodic=false, model=p)
    X = features(atoms, positions, lattice, Ecut, periodic)

    Z=atoms
    Zv=map(Z) do z
        elements[z].shells[end]
    end

    ρv = sum(p .* X)
    ρv = max.(0, ρv)

    dv=det(lattice)/length(ρv)
    rescale(ρv, sum(Zv)/dv)
end

function train(data;nsamples=5000)

    # function train(atoms,positions,lattice,ρv;Ecut=40,periodic=false,nsamples=5000)
    data_train = []
    for case in data
        @unpack Z, positions, lattice, ρv, Ecut,  = case
        periodic=false
       
        X = features(Z, positions, lattice, Ecut, periodic)
        y = ρv

        N = length(Z)
        n = N * nsamples
        Random.seed!(1)
        ix = sample(1:length(ρv), n)

        A = [X[i][j] for i in eachindex(X), j in ix]'
        b = y[ix]

        t = (; A, b)
        # t=(;dx,A0, X0, b,ρv_SAD,ρc_SAD,ρv)
        push!(data_train, t)
    end
    A = vcat(getproperty.(data_train, :A)...)
    b = vcat(getproperty.(data_train, :b)...)
    # ρc_SAD= getindex.(data, 3)
    # X0= getindex.(data, 4)
    # ρc_SAD= getindex.(data, 6)

    @show size(A)
    p = A \ b

    @show nae(A * p, b)
    p
end
