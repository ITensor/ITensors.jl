@eval module $(gensym())
using NDTensors

include(joinpath(pkgdir(NDTensors), "src", "SetParameters", "test", "runtests.jl"))
end
