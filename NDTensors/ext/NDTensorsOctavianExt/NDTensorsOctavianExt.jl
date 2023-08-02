module NDTensorsOctavianExt

using NDTensors

if isdefined(Base, :get_extension)
  using Octavian
else
  using ..Octavian
end

include("import.jl")
include("octavian.jl")

end
