module TypeParameterAccessors
include("undefinedposition.jl") 
include("parameters.jl")
include("unspecify_parameters.jl")

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

export Position,
  TypeParameter,
  default_parameter,
  default_parameters,
  parameter,
  parameters,
  nparameters,
  set_eltype,
  set_ndims,
  set_parameter,
  set_parameters,
  specify_default_parameters,
  specify_parameter,
  unspecify_parameters,
  is_wrapped_array,
  parenttype
end # module
