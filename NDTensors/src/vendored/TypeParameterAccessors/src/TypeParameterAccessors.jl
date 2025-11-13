module TypeParameterAccessors

# Exports
export type_parameters, get_type_parameters
export nparameters, is_parameter_specified
export default_type_parameters
export set_type_parameters, set_default_type_parameters
export specify_type_parameters, specify_default_type_parameters
export unspecify_type_parameters

# Imports
using SimpleTraits: SimpleTraits, @traitdef, @traitimpl

include("type_utils.jl")
include("type_parameters.jl")

# Implementations
include("ndims.jl")
include("base/abstractarray.jl")
include("base/similartype.jl")
include("base/linearalgebra.jl")

end
