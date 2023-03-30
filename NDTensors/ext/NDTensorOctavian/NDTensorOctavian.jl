module NDTensorOctavian

using NDTensors

if isdefined(Base, :get_extension)
  using Octavian
else
  using ..Octavian
end
println("Using octavian")

include("import.jl")
include("octavian.jl")

end
