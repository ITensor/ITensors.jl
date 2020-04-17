using Documenter, ITensors

DocMeta.setdocmeta!(ITensors,
                    :DocTestSetup,
                    :(using ITensors);
                    recursive=true)

makedocs(sitename = "ITensors.jl";
         modules = [ITensors],
         pages = ["Introduction" => "index.md",
                  "Index" => "IndexType.md",
                  "IndexSet" => "IndexSetType.md",
                  "ITensor" => "ITensorType.md",
                  "MPS and MPO" => "MPSandMPO.md",
                  "DMRG" => "DMRG.md"],
         format = Documenter.HTML(prettyurls = false),
         doctest = true,
         checkdocs = :none)
