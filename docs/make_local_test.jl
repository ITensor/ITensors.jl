include("settings.jl")

makedocs(sitename = sitename;
         modules = modules,
         pages = pages,
         format = Documenter.HTML(prettyurls = false),
         doctest = true,
         checkdocs = :none)
