using Documenter

include("../src/NeuralDFT.jl")
using .NeuralDFT

# include("../src/operators.jl")

##
makedocs(
    sitename = "NeuralDFT.jl",
    pages = ["index.md","guide.md","tutorials"=>["install.md","basics.md","dft.md",]]
    )
    # pages = ["index.md", "architecture.md", "publications.md", "tutorials.md"],
