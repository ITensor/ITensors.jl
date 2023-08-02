module NDTensorsTBLISExt

using NDTensors
using LinearAlgebra
if isdefined(Base, :get_extension)
  using TBLIS
else
  using ..TBLIS
end
isdefined(Base, :get_extension) ? (using TBLIS) : (using ..TBLIS)

import NDTensors.contract!

include("contract.jl")
end
