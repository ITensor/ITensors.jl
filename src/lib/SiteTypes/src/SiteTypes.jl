module SiteTypes
# TODO: This is a bit strange, but required for backwards
# compatibility since `val` is also used by `QNVal`.
import ..ITensors: val
# TODO: Use explicit overloading with `NDTensors.space`.
import ..ITensors: space
include("sitetype.jl")
include("SiteTypesChainRulesCoreExt.jl")
include("sitetypes/aliases.jl")
include("sitetypes/generic_sites.jl")
include("sitetypes/qubit.jl")
include("sitetypes/spinhalf.jl")
include("sitetypes/spinone.jl")
include("sitetypes/fermion.jl")
include("sitetypes/electron.jl")
include("sitetypes/tj.jl")
include("sitetypes/qudit.jl")
include("sitetypes/boson.jl")
end
