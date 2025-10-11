using AMDGPU: AMDGPU, ROCArray
using GPUArraysCore: @allowscalar
using NDTensors.Expose: Exposed, expose, parent, unexpose
using NDTensors.GPUArraysCoreExtensions: cpu

function Base.getindex(E::Exposed{<:ROCArray})
    return @allowscalar unexpose(E)[]
end

function Base.setindex!(E::Exposed{<:ROCArray}, x::Number)
    @allowscalar unexpose(E)[] = x
    return unexpose(E)
end

function Base.getindex(E::Exposed{<:ROCArray, <:Adjoint}, i, j)
    return (expose(parent(E))[j, i])'
end

Base.any(f, E::Exposed{<:ROCArray, <:NDTensors.Tensor}) = any(f, data(unexpose(E)))

function Base.print_array(io::IO, E::Exposed{<:ROCArray})
    return Base.print_array(io, expose(cpu(E)))
end
