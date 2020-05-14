using Documenter, ITensors

DocMeta.setdocmeta!(ITensors,
                    :DocTestSetup,
                    :(using ITensors);
                    recursive=true)

makedocs(sitename = "ITensors.jl";
         modules = [ITensors],
         pages = ["Introduction" => "index.md",
                  "Index" => "IndexType.md",
                  "IndexVal" => "IndexValType.md",
                  "IndexSet" => "IndexSetType.md",
                  "ITensor" => "ITensorType.md",
                  "MPS and MPO" => "MPSandMPO.md",
                  "DMRG" => "DMRG.md",
                  "AutoMPO" => "AutoMPO.md",
                  "ProjMPO" => "ProjMPO.md"]
         format = Documenter.HTML(prettyurls = false),
         doctest = true,
         checkdocs = :none)

deploydocs(repo = "github.com/ITensor/ITensors.jl.git")

