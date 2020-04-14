using Documenter, ITensors

makedocs(sitename = "ITensors.jl";
         modules = [ITensors],
         pages = ["Introduction" => "index.md",
                  "Index" => "Index.md",
                  "IndexSet" => "IndexSet.md",
                  "ITensor" => "ITensor.md",
                  "MPS and MPO" => "MPSandMPO.md",
                  "DMRG" => "DMRG.md"],
         format = Documenter.HTML(prettyurls = false),
         doctest = true,
         checkdocs = :none)
