using UnPack
using FileIO
using LinearAlgebra
# using GLMakie
using Random
using StatsBase
using DataStructures
using PeriodicTable
using Combinatorics
# using EquivariantOperators
include("../../EquivariantOperators.jl/src/operators.jl")
include("utils.jl")
Random.seed!(1)

const p = load("model.jld2", "p")

function rescale(a, s)
    a / sum(a) * s
end

function mix(X,n;replace=false)
    # n = length(X)
    # for i = 1:n
    #     for j = i:n
    #         push!(X, X[i] .* X[j])
    #     end
    # end
    f=replace ? with_replacement_combinations : combinations
    map(f(X,n)) do a
        reduce((x,y)->x.*y,a)
    end
end

# function predict_density(Z,pos,lattice,resolution,periodic)
function features(Z, pos,origin, lattice, sz, periodic)
    cell = lattice ./ sz
    grid = Grid(cell, origin,sz; )
    @unpack dv = grid

    Zv = broadcast(x ->
            elements[x].shells[end], Z)
    Zc = Z - Zv
    zpos = DefaultDict(Vector)
    zposc = DefaultDict(Vector)
    # X = OrderedDict()
    X =[]
    ρzv=zeros(sz)

    for (z, zc, p) in zip(Z, Zc, eachcol(pos))
        push!(zpos[z], p)
        push!(zposc[zc], p)
    end
    for z in [1,6,7,8,9]
        if z in Z
        ρ = zeros(sz)
        zv=valence(z)
        zc=core(z)
        ρzv+=zv*ρ
        for p in zpos[z]
            put!(ρ,grid, p, 1)
        end

        pad=:same
        rmax = 6.0
        border = periodic ? :circular : 0
        alg = :fft

        σ = atom_decay_length(z)
        gev = zv * Gaussian(cell, σ, 3σ; pad, border, alg)(ρ)
        e = Op(r -> exp(-r/σ), rmax, cell; pad, border, alg)
        dev=zv*e(ρ)
        
        if zc>0
            σ = atom_decay_length(zc)
            gec = zc * Gaussian(cell, σ, 3σ; pad, border, alg)(ρ)
            dec=zc*Op(r -> exp(-r/σ), rmax, cell; pad, border, alg)(ρ)
        else
            gec=zeros(sz)
            dec=zeros(sz)
        end

        rmin = 1e-9
        ϕ = Op(r -> 1 / (4π * r), rmax, cell; rmin, pad, border, alg)
        
        ϕ0 = ϕ(ρ)
        ϕz = z * ϕ0
        ϕzc = zc * ϕ0
        
        ϕgev = ϕ(gev)
        ϕgec = ϕ(gec)
        ϕdec = ϕ(dec)
        ϕdev = ϕ(dev)
        # ϕ


        # X[z,:] = (; gev, gec, ϕz, ϕzc, ϕgev, ϕgec)
        push!(X,[ gev, gec, dev,dec,ϕz, ϕzc, ϕgev, ϕgec,ϕdec,ϕdev])
    else
        push!(X,fill(zeros(sz),10))
        # gev= gec= ϕz= ϕzc= ϕgev= ϕgec=zeros(sz)
    end
    end
    X=reduce(hcat,X)

    N = sum(Z)
    Nv = sum(Zv)
    Nc = sum(Zc)
    @assert N == Nv + Nc
    # @show sum(gev) * dv, Nv
    # @show sum(gec) * dv, Nc


    # X0 = reduce(vcat, collect.(values(X)))
    # X0 = [sum(getproperty.(values(X),k)) for k in sort(collect(keys(first(values(X)))))]
    # mix!(X0)
    
    # mix!(X)
    # X = [X0..., [op(ρev) for op in ops[[]]]...]
    # X0
    
    F=[sqrt,x->x^2]
    X=sum(X,dims=2)
    X=reduce(vcat,[mix(X,n) for n=1:3])

    X_=map(F) do f
        map(X) do x
            f.(abs.(x))
            # f.(x)
        end
    end
    X=vcat(X,reduce(vcat,X_))
    (;X,ρzv,cell)
end

function predict_density(Z, pos, lattice,resolution::Real;kw...)
    sz = norm.(eachcol(lattice)) .÷ resolution
     predict_density(Z, pos, lattice,sz;kw...)
end

function predict_density(Z, pos, lattice,sz; periodic=false, model=p,origin=ones(3))
# function predict_density(Z, pos, lattice,sz; periodic=false, model=p,origin=(sz.+1)./2)
  @unpack ρzv,cell,  X = features(Z, pos,origin, lattice, sz, periodic)

    Zv = map(Z) do z
        elements[z].shells[end]
    end

    ρev = sum(p .* X)
    ρev = max.(0, ρev)

    dv = det(lattice) / length(ρev)
    ρev= rescale(ρev, sum(Zv) / dv)

    if verbose
        return (;Z,pos,lattice,cell,ρev,ρzv, Zv,periodic,origin)
    end
    return ρev
end

function train(data; nsamples=5000)
    data_train = []
    for case in data
        @unpack Z, pos, lattice, ρev = case
        sz=size(ρev)

        origin=ones(3)
        periodic = false

        X = features(Z, pos,origin, lattice,sz, periodic)
        y = ρev

        N = length(Z)
        n = N * nsamples
        Random.seed!(1)
        ix = sample(1:length(ρev), n)

        A = [X[i][j] for i in eachindex(X), j in ix]'
        b = y[ix]

        t = (; A, b)
        # t=(;dx,A0, X0, b,ρev_SAD,ρc_SAD,ρev)
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
