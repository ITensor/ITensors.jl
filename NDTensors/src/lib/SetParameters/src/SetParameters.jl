module SetParameters
include("position.jl")
include("unspecifiedparameter.jl")
include("default_parameter.jl")
# include("interface.jl")
include("get_parameter.jl")
# include("set_parameters_generic.jl")
# include("set_parameters.jl")
include("specify_parameters.jl")
# include("set_default_parameters.jl")
# include("Base/val.jl")
include("Base/abstractarray.jl")
# include("Base/array.jl")
# include("Base/subarray.jl")

## TODO when this is a full package utilize this function to
# # enable extensions
# using PackageExtensionCompat
# function __init__()
#   @require_extensions
# end

# include("../ext/SetParametersFillArraysExt/SetParametersFillArraysExt.jl")
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
