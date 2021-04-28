include("settings.jl")

settings[:doctest] = false

makedocs(; sitename=sitename, settings...)
