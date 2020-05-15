using Documenter, ITensors

DocMeta.setdocmeta!(ITensors,
                    :DocTestSetup,
                    :(using ITensors);
                    recursive=true)

sitename = "ITensors.jl"

settings = Dict(
  :modules => [ITensors],
  :pages => [
          "Introduction" => "index.md",
          "Index" => "IndexType.md",
          "IndexVal" => "IndexValType.md",
          "IndexSet" => "IndexSetType.md",
          "ITensor" => "ITensorType.md",
          "MPS and MPO" => "MPSandMPO.md",
          "QN" => "QN.md",
          "DMRG" => "DMRG.md",
          "AutoMPO" => "AutoMPO.md",
          "ProjMPO" => "ProjMPO.md",
          "ProjMPOSum" => "ProjMPOSum.md",
           ],
  :format => Documenter.HTML(prettyurls = false),
  :doctest => true,
  :checkdocs => :none,
 )
