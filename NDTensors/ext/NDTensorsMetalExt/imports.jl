import NDTensors: mtl, set_ndims, set_eltype
import NDTensors.SetParameters: nparameters, get_parameter, set_parameter, default_parameter

using NDTensors.Unwrap: Exposed, unwrap_type, unexpose, expose
using Metal: DefaultStorageMode
using NDTensors: adapt
