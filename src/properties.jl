include("density.jl")

struct ForceField
    cell
    op
    function ForceField(cell::AbstractMatrix; rmax = 6.0)
        @info "Instantiating `ForceField` at discretisation cell $cell. Calculating convolution kernel arrays. This can take a minute..."
        
        pad=:same
        l=1
        rmin=2maximum(norm.(eachcol(cell)))-1e-6
        op = Op(r -> 1 / (r^2), rmax, cell;l, rmin, pad)
        new(cell,op)
    end
end

"""
    ForceField(cell)
    (m::ForceField)(Z, pos, ρ,; [origin],[cell], [periodic]  )

Constructs `ForceField` which is callable for computing forces when given valence electron density. It instantiates kernels at the specified discretisation setting and computes force fields via FFT. discretisation at construction and usage can be different. if discretisation of the valence density is different from that of `ForceField` functor, the former must be supplied as a keyword `cell` when calling the latter

# Args
- `cell`: column-wise matrix of discretization cell vectors
- `Z`: list of atomic numbers
- `pos`: column wise position matrix of atoms in `Z`. size of 3 x # electrons.
- `ρ`: valence electron density
- `periodic`: set to `true` for crystal lattices / periodic boundaries
- `origin`: indices of the origin when specifying positions, may be decimal valued
"""
function (m::ForceField)(Z,pos,ρ,;origin=ones(3),cell=m.cell, periodic=false,  )
    ρ=imresize(-ρ,Tuple(round.(Int,norms(cell).*size(ρ)./norms(m.cell))))
    sz=size(ρ)
    if origin == :center
        origin = (1 .+ sz) ./ 2
    end
    
    g=Grid(m.op.grid.cell,origin)
    for (z,p) in zip(Z,eachcol(pos))
        ρ[g, p...]=valence(z)
    end
E=m.op(ρ;periodic)
        reduce(hcat,[z*E[g,I...] for (I,z) in zip(eachcol(pos),Z)])
end