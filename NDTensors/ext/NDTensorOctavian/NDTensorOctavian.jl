module NDTensorOctavian

using NDTensors
using LinearAlgebra

if isdefined(Base, :get_extension)
    using Octavian
else
    using ..Octavian
end
println("Using octavian")

include("import.jl")
include("octavian.jl")


#export backend_octavian
end