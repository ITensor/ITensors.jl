module TypeParameterAccessors
include("Base/internals.jl")
include("position.jl")
include("parameters.jl")
include("self.jl")
include("interface.jl")

include("set_parameters.jl")
include("specify_parameters.jl")
include("default_parameters.jl")
include("Base/iswrappedarray.jl")
include("Base/array.jl")

export Position,
  nparameters,
  parameter,
  parameters,
  parenttype,
  position,
  position_name,
  position_names,
  set_eltype,
  set_ndims,
  set_parameter,
  set_parameters,
  specify_parameter,
  specify_parameters,
  set_default_parameter,
  set_default_parameters,
  specify_default_parameter,
  specify_default_parameters,
  unwrap_array_type
end
