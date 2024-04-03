# I tried putting this inside of an
# `NDTensorsNamedDimsArraysExt` module
# but for some reason it kept overloading
# `Base.similar` instead of `NDTensors.similar`.
include("similar.jl")
include("fill.jl")
