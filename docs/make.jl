using ITensors: ITensors
using Documenter: Documenter, DocMeta, deploydocs, makedocs

DocMeta.setdocmeta!(ITensors, :DocTestSetup, :(using ITensors); recursive=true)

include("make_index.jl")

makedocs(;
  modules=[ITensors],
  authors="ITensor developers <support@itensor.org> and contributors",
  sitename="ITensors.jl",
  format=Documenter.HTML(;
    canonical="https://itensor.github.io/ITensors.jl",
    edit_link="main",
    assets=["assets/favicon.ico", "assets/extras.css"],
  ),
  pages=["Home" => "index.md", "Reference" => "reference.md"],
)

deploydocs(; repo="github.com/ITensor/ITensors.jl", devbranch="main", push_preview=true)
