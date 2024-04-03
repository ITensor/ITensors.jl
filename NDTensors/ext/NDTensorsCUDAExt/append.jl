using GPUArraysCore: @allowscalar
using CUDA: CuArray
using NDTensors.Expose: Exposed, unexpose

## Warning this append function uses scalar indexing and is therefore extremely slow
function Base.append!(Ecollection::Exposed{<:CuArray}, collections...)
  return @allowscalar append!(unexpose(Ecollection), collections...)
end
