using GPUArraysCore: @allowscalar
using CUDA: CuArray
using NDTensors.Expose: Exposed, unexpose

function Base.append!(Ecollection::Exposed{<:CuArray}, collections...)
  return @allowscalar append!(unexpose(Ecollection), collections...)
end
