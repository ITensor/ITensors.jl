## Right now append! is broken on metal but make this available for when it is working
using GPUArraysCore: @allowscalar
using Metal: MtlArray
using NDTensors.Expose: Exposed, unexpose

function Base.append!(Ecollection::Exposed{<:MtlArray}, collections...)
  return @allowscalar append!(unexpose(Ecollection), collections...)
end
