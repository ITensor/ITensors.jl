using Metal: MtlArray
using GPUArraysCore: @allowscalar
using LinearAlgebra: Adjoint
using NDTensors.Expose: Exposed, expose, unexpose

function Base.getindex(E::Exposed{<:MtlArray})
    return @allowscalar unexpose(E)[]
end

function Base.setindex!(E::Exposed{<:MtlArray}, x::Number)
    @allowscalar unexpose(E)[] = x
    return unexpose(E)
end

# Shared with `CuArray`. Move to `NDTensorsGPUArraysCoreExt`?
function Base.getindex(E::Exposed{<:MtlArray, <:Adjoint}, i, j)
    return (expose(parent(E))[j, i])'
end
