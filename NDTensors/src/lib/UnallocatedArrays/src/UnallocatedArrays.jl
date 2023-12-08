module UnallocatedArrays
using FillArrays:
  FillArrays,
  AbstractFill,
  AbstractZeros,
  Fill,
  Zeros,
  broadcasted_zeros,
  getindex_value,
  kron_zeros,
  mult_zeros
using NDTensors.SetParameters:
  SetParameters,
  Position,
  default_parameter,
  get_parameter,
  nparameters,
  set_parameter,
  set_parameters

include("abstractfill/abstractfill.jl")
include("abstractfill/set_types.jl")

include("unallocatedfill.jl")
include("unallocatedzeros.jl")
include("abstractunallocatedarray.jl")
include("set_types.jl")

export UnallocatedFill, UnallocatedZeros, alloctype, set_alloctype, allocate
end