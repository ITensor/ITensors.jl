## Right now append! is broken on metal because of a missing resize! function
## but make this available in the next release this will allow metal to work working
using GPUArraysCore: @allowscalar
using Metal: MtlArray
using NDTensors.Expose: Exposed, unexpose

## Warning this append function uses scalar indexing and is therefore extremely slow
function Base.append!(Ecollection::Exposed{<:MtlArray}, collections...)
    return @allowscalar append!(unexpose(Ecollection), collections...)
end
