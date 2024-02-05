module SetParameters
include("position.jl")
include("unspecifiedparameter.jl")
include("default_parameter.jl")
include("get_parameter.jl")
include("specify_parameters.jl")
# include("set_default_parameters.jl")
include("Base/abstractarray.jl")

## TODO when this is a full package utilize this function to
# # enable extensions
# using PackageExtensionCompat
# function __init__()
#   @require_extensions
# end

# This implementation relies on Julia Base internals
include("typeparameteraccessor.jl")

export DefaultParameter,
  DefaultParameters,
  Position,
  default_parameter,
  get_parameter,
  parameters,
  nparameters,
  set_eltype,
  set_ndims,
  set_parameters,
  specify_parameters
end # module
