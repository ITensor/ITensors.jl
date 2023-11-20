module NDTensorsTestUtils

using NDTensors

include("device_list.jl")
include("is_suppoted_eltype.jl")

default_rtol(elt::Type) = 10^(0.75 * log10(eps(real(elt))))

export default_rtol, is_supported_eltype, devices_list
end
