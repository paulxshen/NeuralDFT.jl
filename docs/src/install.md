# Installation

- Install Julia, VisualStudio
- `git clone https://github.com/paulxshen/NeuralDFT.jl`
- open folder in VisualStudio
- start Julia REPL
- `]instantiate` to install core dependencies from Project.toml
- `]add UnPack,FileIO,JLD2,PeriodicTable,DataStructures,EquivariantOperators,Statistics,GLMakie` for tutorials
    - if `GLMakie`: GPU plotting fails to install, just skip plotting by commenting out `GLMakie`, `volume(...)` lines. `Plots.plot` can also plot 3d volumes but it can't be rotated
- try running tutorial `example/basics.jl`

## Optional
- some tutorials use additional packages which can be installed via:
    - `]add DFTK@0.3.10`: has Installation issues - not needed unless you're running comparisons vs DFT. we use an older version of DFTK.jl (planewaves DFT package) because I'm unable to get the newest version to build correctly.
- 

[](<!--- `]add DFTK` or  -->)

In case of errors, I'm probably the one that screwed up. slack/email me a screenshot of your life's technical difficulties so I can help :)