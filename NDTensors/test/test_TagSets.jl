@eval module $(gensym())
using NDTensors

include(joinpath(pkgdir(NDTensors), "src", "TagSets", "test", "runtests.jl"))
end