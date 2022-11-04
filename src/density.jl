using UnPack
using FileIO
using LinearAlgebra
# using GLMakie
using Random
using StatsBase
using DataStructures
using PeriodicTable
using Combinatorics
# include("../../EquivariantOperators.jl/src/EquivariantOperators.jl")
using EquivariantOperators
# include("../../EquivariantOperators.jl/src/operators.jl")
# include("utils.jl")
Random.seed!(1)

SAD = load("SAD1025.jld2", "SAD",)
valence(z)=elements[z].shells[end]
core(z)=z-valence(z)


function mix(X, n; replace=false)
    # n = length(X)
    # for i = 1:n
    #     for j = i:n
    #         push!(X, X[i] .* X[j])
    #     end
    # end
    f = replace ? with_replacement_combinations : combinations
    map(f(X, n)) do a
        reduce((x, y) -> x .* y, a)
    end
end

function calczops(z, cell)
    @unpack R, dr = SAD[z]
    n = length(R)
    g = Grid((dr,))
    R = vcat(reverse(R[2:end]), R)
    pad = :same
    m = hcat([
        R,
        R .^ 2,
        sqrt.(abs.(R)),
        Del(g.cell)(R),
        # Laplacian(g.cell)(R),
        Op(r -> exp(-r), 4, g.cell; pad)(R),
        Op(r -> exp(-(r)^2), 2, g.cell; pad)(R),
    ]...)
    global m = m[(length(R)+1)÷2:end, :]
    rmax = 0.99 * dr * (n - 1)
    global f = [Op(r -> a[g, r], rmax, cell) for a in eachcol(m)]

    # for f in f
    # f.kernel = rescale(f.kernel, 1)
end

mutable struct DensityModel
    Z
    p
end
"""
"""


struct DensityPredictor
    cell::AbstractMatrix
    # periodic::Bool
    zops::AbstractDict
    # ops
    model
end


# - `domain`: 3 x 3 matrix with columns as bounding Vectors for the domain.

"""
    DensityPredictor(cell, model)
    DensityPredictor(domain, resolution, model)
    (m::DensityPredictor)(Z, pos, sz; periodic, origin)

Constructs `DensityPredictor` which is callable for Predicting valence electron density. It instantiates equivariant kernels at the specified discretisation setting and computes the prediction using `DensityModel` parameters. 

# Args
- `cell`: column-wise matrix of discretization cell vectors
- `Z`: list of atomic numbers
- `pos`: column wise position matrix of atoms in `Z`. size of 3 x # electrons.
- `sz`: integer size (pixel dimensions) of domain grid
- `periodic`: set to `true` for crystal lattices / periodic boundaries
- `origin`: indices of the origin when specifying positions, may be decimal valued
- `domain`: 3 x 3 matrix with columns as bounding Vectors for the domain.
- `resolution`: discretisation length
"""
function DensityPredictor(cell::AbstractMatrix, model=nothing)
    @info "Instantiating `DensityPredictor` at discretisation cell $cell. Calculating convolution kernel arrays. This can take a minute..."
    zops = Dict([z => calczops(z, cell) for z in model.Z])
    # zops = hcat(map(z->calczops(z, cell),[1, 6, 7, 8])...)
    DensityPredictor(cell, zops, model)
end

function DensityPredictor(domain::AbstractMatrix, resolution::Real,model=nothing)
    sz = round.(Int, norm.(eachcol(domain)) ./ resolution)
    cell=domain./sz
    DensityPredictor(cell,model)
end
using ImageTransformations


function (m::DensityPredictor)(Z::Base.AbstractVecOrTuple, pos::AbstractMatrix,domain::AbstractMatrix,sz::Base.AbstractVecOrTuple;kw...)
    r=m(Z,pos,domain;kw...,mode=:verbose)
    @unpack ρ,ρzv=r
    ρ=imresize(ρ,sz)
    ρzv=imresize(ρzv,sz)
    grid=Grid(domain./sz,origin)
# (;r...,ρ,ρzv,grid)
ρ
end
function (m::DensityPredictor)(Z::Base.AbstractVecOrTuple, pos::AbstractMatrix,domain::AbstractMatrix;kw...)
    sz = round.(Int, norm.(eachcol(domain)) ./ norm.(eachcol(m.cell)))
    m(Z,pos,sz;kw...)
end

function (m::DensityPredictor)(Z::Base.AbstractVecOrTuple, pos::AbstractMatrix, sz::Base.AbstractVecOrTuple; periodic=false, origin=ones(3), stage=nothing,mode=nothing)
    Zv=valence.(Z)
    sz=Tuple(sz)
    if origin == :center
        origin = (1 .+ sz) ./ 2
    end
    @show origin

    @unpack model, zops, cell = m
    grid = Grid(cell, origin, ;)
    @unpack dv = grid

    zpos = DefaultDict(Vector)

    pad = :same
    border = periodic ? :circular : 0

    ρzv = zeros(sz)
    X = []
    for (z, p) in zip(Z, eachcol(pos))
        push!(zpos[z], p)
    end

    X=map( model.Z) do z
        if z in Z
            ρi = zeros(sz)
            zv = valence(z)
            zc = core(z)
            for p in zpos[z]
                put!(ρi, grid, p, 1)
            end
            ρzv += zv * ρi
          r=  map(zops[z]) do op
                op(ρi; border, pad) end
                # delete!(cache,objectid(ρi))
                return r
        else
            return fill(zeros(sz), length(zops[1]))
        end
    end
    X = hcat(X...)
    X = reduce(vcat, [mix(X, n; replace=true) for n = 1:2])

    if stage==:conv
return    (; X, ρzv, grid)
    end


    p = model.p
    ρ = sum(p .* X)
    ρ = max.(0, ρ)
    ρ = rescale(ρ, sum(Zv) / grid.dv)
    # dv = det(domain) / length(ρ)
if mode==:verbose
        return (; Z, pos,  grid, ρ, ρzv, Zv, periodic, origin, )
end
ρ
        # return (; Z, pos, domain, grid, ρ, ρzv, Zv, periodic, origin, X)
end

"""
    train(data; nsamples=100)

Trains a `DensityModel` from data

# Args
- `nsamples`: # samples / cubic bohr
- `data`: list of cases each of which has the fields:
    - `Z`: list of atomic numbers
    - `pos`: column wise position matrix of atoms in `Z`. size of 3 x # electrons.
    - `sz`: integer size (pixel dimensions) of domain grid
    - `periodic`: set to `true` for crystal lattices / periodic boundaries
    - `domain`: 3 x 3 matrix with columns as bounding Vectors for the domain.
    - `origin`: indices of the origin when specifying positions, may be decimal valued
"""
function train(data; nsamples=100)
    data_train = []
    @unpack domain, origin, ρ = data[1]
    sz = size(ρ)
    cell = domain ./ sz
    dv = det(cell)
    # @unpack cell = data[1]

    Z = sort(unique(reduce(vcat, getfield.(data, :Z))))
    p = nothing
    model = DensityModel(Z, p)
    m = DensityPredictor(cell, model)

    # empty!(cache)
    for case in data
        @unpack Z, pos, domain, origin, ρ, periodic = case

        @unpack X = m(Z, pos,  sz,;origin, periodic,stage=:conv)
        y = ρ

        n = round(Int, dv * prod(sz) * nsamples)
        Random.seed!(1)
        ix = sample(1:length(ρ), n)

        A = [X[i][j] for i in eachindex(X), j in ix]'
        b = y[ix]

        t = (; A, b)
        # t=(;dx,A0, X0, b,ρ_SAD,ρc_SAD,ρ)
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
    m.model.p = p
    m
end
