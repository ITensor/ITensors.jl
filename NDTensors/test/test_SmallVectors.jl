@eval module $(gensym())
using NDTensors

include(joinpath(pkgdir(NDTensors), "src", "SmallVectors", "test", "runtests.jl"))
end