module NDTensorsTestUtils

using NDTensors

include("device_list.jl")
include("is_supported_eltype.jl")

default_rtol(elt::Type) = 10^(0.75 * log10(eps(real(elt))))

end
