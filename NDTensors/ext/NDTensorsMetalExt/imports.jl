import NDTensors: mtl, set_ndims, set_eltype, set_eltype_if_unspecified
import NDTensors.SetParameters: nparameters, get_parameter, set_parameter, default_parameter
import NDTensors.Unwrap: Exposed

using Metal: DefaultStorageMode
using NDTensors: adapt
