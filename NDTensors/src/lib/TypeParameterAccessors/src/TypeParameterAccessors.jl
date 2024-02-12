module TypeParameterAccessors
include("parameters.jl")

include("undefpositon.jl")
include("position.jl")
include("interface.jl")

include("set_parameters.jl")

include("specify_parameters.jl")
include("default_parameter.jl")

include("Base/iswrappedarray.jl")
include("Base/abstractarray.jl")

## TODO when this is a full package utilize this function to
# # enable extensions
# using PackageExtensionCompat
# function __init__()
#   @require_extensions
# end

# This implementation relies on Julia Base internals
include("typeparamaccessor.jl")

export DefaultParameter,
  DefaultParameters,
  Position,
  TypeParameter,
  default_parameter,
  parameter,
  parameters,
  nparameters,
  set_eltype,
  set_ndims,
  set_parameter,
  set_parameters,
  specify_parameter,
  is_wrapped_array,
  parenttype
end # module
