@eval module $(gensym())
using NDTensors

include(joinpath(pkgdir(NDTensors), "src", "NamedDimsArrays", "test", "runtests.jl"))
end