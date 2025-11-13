using JLArrays: JLArray
using GPUArraysCore: @allowscalar
using NDTensors: NDTensors
using NDTensors.Expose: Exposed, expose, unexpose

function Base.getindex(E::Exposed{<:JLArray})
    return @allowscalar unexpose(E)[]
end

function Base.setindex!(E::Exposed{<:JLArray}, x::Number)
    @allowscalar unexpose(E)[] = x
    return unexpose(E)
end

function Base.getindex(E::Exposed{<:JLArray, <:Adjoint}, i, j)
    return (expose(parent(E))[j, i])'
end

Base.any(f, E::Exposed{<:JLArray, <:NDTensors.Tensor}) = any(f, data(unexpose(E)))
