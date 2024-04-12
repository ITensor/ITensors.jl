using GPUArraysCore: @allowscalar
using AMDGPU: ROCArray
using NDTensors.Expose: Exposed, unexpose

## Warning this append function uses scalar indexing and is therefore extremely slow
function Base.append!(Ecollection::Exposed{<:ROCArray}, collections...)
  return @allowscalar append!(unexpose(Ecollection), collections...)
end
