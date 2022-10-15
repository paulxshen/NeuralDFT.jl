using DFTK

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

kgrid = [1, 1, 1]

const EN = Dict([
    1 => 2.2,
    6 => 2.55,
    7 => 3.04,
    8 => 3.44,
    9 => 3.98,
    14 => 1.9,
    16 => 2.58,
    17 => 3.16,
])

valence(z)=elements[z].shells[end]
core(z)=z-valence(z)

function atom_decay_length(z)
    atom_decay_length(core(z),valence(z))
end

function atom_decay_length(n_elec_core, n_elec_valence)
    # Adapted from ABINIT/src/32_util/m_atomdata.F90,
    # from which also the data has been taken.

    n_elec_valence = round(Int, n_elec_valence)
    if n_elec_valence == 0
        return 0.0
    end

    data = if n_elec_core < 0.5
        # Bare ions: Adjusted on 1H and 2He only
        [0.6, 0.4, 0.3, 0.25, 0.2]
    elseif n_elec_core < 2.5
        # 1s2 core: Adjusted on 3Li, 6C, 7N, and 8O
        [1.8, 1.4, 1.0, 0.7, 0.6, 0.5, 0.4, 0.35, 0.3]
    elseif n_elec_core < 10.5
        # Ne core (1s2 2s2 2p6): Adjusted on 11na, 13al, 14si and 17cl
        [2.0, 1.6, 1.25, 1.1, 1.0, 0.9, 0.8, 0.7 , 0.7, 0.7, 0.6]
    elseif n_elec_core < 12.5
        # Mg core (1s2 2s2 2p6 3s2): Adjusted on 19k, and on n_elec_core==10
        [1.9, 1.5, 1.15, 1.0, 0.9, 0.8, 0.7, 0.6 , 0.6, 0.6, 0.5]
    elseif n_elec_core < 18.5
        # Ar core (Ne + 3s2 3p6): Adjusted on 20ca, 25mn and 30zn
        [2.0, 1.8, 1.5, 1.2, 1.0, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.65, 0.6]
    elseif n_elec_core < 28.5
        # Full 3rd shell core (Ar + 3d10): Adjusted on 31ga, 34se and 38sr
        [1.5, 1.25, 1.15, 1.05, 1.00, 0.95, 0.95, 0.9, 0.9, 0.85, 0.85, 0.80,
         0.8 , 0.75, 0.7]
    elseif n_elec_core < 36.5
        # Krypton core (Ar + 3d10 4s2 4p6): Adjusted on 39y, 42mo and 48cd
        [2.0, 2.00, 1.60, 1.40, 1.25, 1.10, 1.00, 0.95, 0.90, 0.85, 0.80, 0.75, 0.7]
    else
        # For the remaining elements, consider a function of n_elec_valence only
        [2.0 , 2.00, 1.55, 1.25, 1.15, 1.10, 1.05, 1.0 , 0.95, 0.9, 0.85, 0.85, 0.8]
    end
    data[min(n_elec_valence, length(data))]
end