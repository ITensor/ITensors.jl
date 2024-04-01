using GPUArraysCore: @allowscalar
using AMDGPU: ROCArray
using NDTensors.Expose: Exposed, unexpose

function Base.append!(Ecollection::Exposed{<:ROCArray}, collections...)
  return @allowscalar append!(unexpose(Ecollection), collections...)
end
