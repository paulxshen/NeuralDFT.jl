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
    14=>ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4")),
    16 => ElementPsp(:S, psp=load_psp("hgh/lda/s-q6")),
    17 => ElementPsp(:Cl, psp=load_psp("hgh/lda/Cl-q7")),
])
He=psp[2]
Ne=psp[10]
Si=psp[14]

name = :Si
name = :qm9
if name == :qm9
    cases = []
    folder = "data\\qm9"
    files = collect(readdir(folder))
    for file in files[1:4]
        p = "$folder\\$file"
        header, atom_labels, geoms = read_xyz(p)

        # for (x, atoms, geom) in zip(header, atom_labels, geoms)
        (x, atoms, geom) = first(zip(header, atom_labels, geoms))
        symbols = Symbol.(atoms)
        Z = broadcast(x -> elements[x].number, symbols)
        Zc = broadcast(x -> try
                elements[x].shells[end-1]
            catch e
                0
            end, symbols)
        valence_Z = Z - Zc

        positions = geom * 1.88973
        a = 2 * maximum(abs.(positions)) .+ 4.0
        positions += a / 2 * ones(size(positions))

        lattice = a * Diagonal(ones(3))
        Ecut = 40
        kgrid = [1, 1, 1]

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

        ρv, ρv_SAD = compute_scfres(atoms, lattice; Ecut, kgrid)
        if !isempty(core_atoms)
            core_model = model_LDA(lattice, core_atoms)
            core_basis = PlaneWaveBasis(core_model; Ecut, kgrid)
            ρc_SAD = guess_density(core_basis)
        else
            ρc_SAD = zeros(size(ρ))
        end
        ρc_SAD = ρc_SAD[:, :, :, 1]
        ρ = ρv + ρc_SAD

        dx = a / size(ρ, 1)
        Zv = Z - Zc
        periodic = false
        case = (; dx, ρ, ρv, ρv_SAD, ρc_SAD, Z, Zc, Zv, positions, lattice, Ecut, periodic)
        push!(cases, case)
    end

    name = "qm9_20"
    name = "small_molecules_v1"
    name = "qm9_4"
    # save("..\\data\\$name.jld2", "cases", cases)
elseif name == :Si
    cases = []

    # 1. Define lattice and atomic positions
    # a = 5.431u"angstrom"          # Silicon lattice constant
    a = 5.431*1.88973
    lattice = a / 2 * [[0 1 1.0]  # Silicon lattice vectors
        [1 0 1.0]  # specified column by column
        [1 1 0.0]]

    # Load HGH pseudopotential for Silicon
    Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))

    # Specify type and positions of atoms
    rvecs=[ones(3) / 8, -ones(3) / 8]
    positions=hcat([lattice*v for v in rvecs]...)
    atoms = [Si => rvecs]
    core_atoms = [Ne => rvecs, He => rvecs]

    # core_atoms = DefaultDict(Vector)
    # for (a, b, x) in zip(Z, Zc, eachcol(positions))
    #     xr = lattice \ x
    #     push!(atoms[psp[a]], xr)
    #     if b != 0
    #         push!(core_atoms[psp[b]], xr)
    #     end
    # end
    # atoms = collect(atoms)
    # core_atoms = collect(core_atoms)

    # 2. Select model and basis
    model = model_LDA(lattice, atoms)
    kgrid = [4, 4, 4]     # k-point grid (Regular Monkhorst-Pack grid)
    Ecut = 40              # kinetic energy cutoff
    # Ecut = 190.5u"eV"  # Could also use eV or other energy-compatible units
    basis = PlaneWaveBasis(model; Ecut, kgrid)
    # Note the implicit passing of keyword arguments here:
    # this is equivalent to PlaneWaveBasis(model; Ecut=Ecut, kgrid=kgrid)

    # 3. Run the SCF procedure to obtain the ground state
    scfres = self_consistent_field(basis, tol=1e-8)
    ρv = scfres.ρ[:, :, :, 1]

    ρv_SAD = guess_density(basis)[:, :, :, 1]
    if !isempty(core_atoms)
        core_model = model_LDA(lattice, core_atoms)
        core_basis = PlaneWaveBasis(core_model; Ecut, kgrid)
        ρc_SAD = guess_density(core_basis)[:, :, :, 1]
    else
        ρc_SAD = zeros(size(ρ))
    end
    ρ = ρv + ρc_SAD

    cell = hcat((eachcol(lattice) ./ size(ρv))...)
    Z = [14, 14]
    Zc = [10, 10]
    Zv = Z - Zc
    periodic = true
    case = (; cell, ρ, ρv, ρv_SAD, ρc_SAD, Z, Zc, Zv, positions, lattice, Ecut, kgrid, periodic)
    push!(cases, case)

    save("..\\data\\$name.jld2", "cases", cases)
end