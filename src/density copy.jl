using UnPack
using FileIO
# using GLMakie
using Random
using StatsBase
using DataStructures
using PeriodicTable
using DFTK
# using EquivariantOperators
include("../../src/operators.jl")

Random.seed!(1)


const psp = Dict([
    1 => ElementPsp(:H, psp=load_psp("hgh/lda/h-q1")),
    2 => ElementPsp(:He, psp=load_psp("hgh/lda/He-q2")),
    3 => ElementPsp(:Li, psp=load_psp("hgh/lda/Li-q3")),
    6 => ElementPsp(:C, psp=load_psp("hgh/lda/c-q4")),
    7 => ElementPsp(:N, psp=load_psp("hgh/lda/n-q5")),
    8 => ElementPsp(:O, psp=load_psp("hgh/lda/o-q6")),
    9 => ElementPsp(:F, psp=load_psp("hgh/lda/f-q7")),
    10 => ElementPsp(:Ne, psp=load_psp("hgh/lda/Ne-q8")),
    16 => ElementPsp(:S, psp=load_psp("hgh/lda/s-q6")),
    17 => ElementPsp(:Cl, psp=load_psp("hgh/lda/Cl-q7")),
])

kgrid=[1, 1, 1]

const EN=Dict([
    1=>2.2,
6=>2.55,
7=>3.04,
8=>3.44,
9=>3.98,
14=>1.9,
16=>2.58,
17=>3.16,
])
    
const p=load("model.jld2","p")

function rescale(a, s)
    a / sum(a) * s
end

    # function predict_density(atoms,positions,lattice,resolution,periodic)
    function features(atoms,positions,lattice,Ecut,periodic)
    # sz=norm.(eachcol(lattice)).÷resolution
    
    Z=atoms
    Zc = broadcast(x -> try
    elements[x].shells[end-1]
catch e
    0
end, symbols)
Zv = Z - Zc

atoms = DefaultDict(Vector)
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
    ρv_SAD= guess_density(basis)
    
    if !isempty(core_atoms)
        core_model = model_LDA(lattice, core_atoms)
        core_basis = PlaneWaveBasis(core_model; Ecut, kgrid)
        ρc_SAD = guess_density(core_basis)
    else
        ρc_SAD = zeros(sz)
    end
    
    ρc_SAD=ρc_SAD[:, :, :, 1]
    ρv_SAD=ρv_SAD[:, :, :, 1]
    
    sz=size(ρv_SAD)
    cell = lattice./sz
    origin = ones(3)
    # origin =periodic ? ones(3) : sz./2
    grid = Grid(cell, sz; origin)
    @unpack dv = grid

    N = sum(Z)
    Nv = sum(Zv)
    Nc = sum(Zc)
    @assert N==Nv+Nc
    @show sum(ρv_SAD) * dv, Nv
    @show sum(ρc_SAD)*dv,Nc
    
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
    border=periodic ? :circular : 0
    alg = :fft
    s = [0.5, 1]
    # s = [0.5, 1, 1.5]
    # s = [1, 2]
    ops = vcat(
        [Op(r -> exp(-r / a), 6a, cell; pad, border,alg) for a = s],
        [Op(r -> exp(-(r / a)^2), 3a, cell; pad, border,alg) for a = s],
        [Op(r -> 1 / r, 6.0, cell; pad, border,alg, rmin=1e-6)],
        )
        
    X0 = [vec([f(u) for f = ops, u = (ρp,  ρpv, ρen,ρc_SAD,ρv_SAD)])..., ρc_SAD,ρv_SAD]
    # X0 = [vec([f(u) for f = ops, u = (ρp,  ρpv, ρen)])..., ρc_SAD,ρv_SAD]
    # X0_ = copy(X0)
    mix!(X0)
    
    # mix!(X)
    # X = [X0..., [op(ρv) for op in ops[[]]]...]
    X0
end

function predict_density(atoms,positions,lattice;Ecut=40,periodic=false,p)
    X=features(atoms,positions,lattice,Ecut,periodic)
    ρv=sum(p.*X)
    ρv=max.(0,ρv)
    rescale(ρv,sum(atoms))
end

function train(data)
    
    # function train(atoms,positions,lattice,ρv;Ecut=40,periodic=false,nsamples=5e3)
    data_train=[]
    for case in data
        @unpack atoms,positions,lattice,ρv,Ecut,periodic = case
        X=features(atoms,positions,lattice,Ecut,periodic)
    y = ρv

    N=length(atoms)
    n=N*nsamples
    Random.seed!(1)
    ix=sample(1:length(ρv),n)

    A = [X[i][j] for i in eachindex(X), j in ix]'
    b = y[ix]

    t=(;A,b)
    # t=(;dx,A0, X0, b,ρv_SAD,ρc_SAD,ρv)
    push!(data_train, t)
    end
A = vcat(getproperty.(data_train, :A)...)
b = vcat(getproperty.(data_train, :b)...)
# ρc_SAD= getindex.(data, 3)
# X0= getindex.(data, 4)
# ρc_SAD= getindex.(data, 6)

@show size(A)
p= A \ b

@show nae(A*p, b)
p
end
