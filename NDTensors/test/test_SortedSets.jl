@eval module $(gensym())
using NDTensors

include(joinpath(pkgdir(NDTensors), "src", "SortedSets", "test", "runtests.jl"))
end
