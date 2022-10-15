include("density.jl")

function predict_forces(ρres    )
    @unpack Z,pos,lattice,cell,ρev,ρzv, Zv,periodic,origin=ρres

    pad=:same
    rmax = 6.0
    border = periodic ? :circular : 0
    l=1
    rmin = .5
        E = Op(r -> 1 / (4π * r^2), rmax, cell;l, rmin, pad, border)(ρev+ρzv)

        g=Grid(cell,origin)
        reduce(hcat,[z*E[g,I...] for (I,z) in zip(eachcol(pos),Z)])
end