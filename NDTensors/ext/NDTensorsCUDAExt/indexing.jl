using CUDA: CuArray
using GPUArraysCore: @allowscalar
using NDTensors: NDTensors
using NDTensors.Expose: Exposed, expose, unexpose

function Base.getindex(E::Exposed{<:CuArray})
    return @allowscalar unexpose(E)[]
end

function Base.setindex!(E::Exposed{<:CuArray}, x::Number)
    @allowscalar unexpose(E)[] = x
    return unexpose(E)
end

function Base.getindex(E::Exposed{<:CuArray, <:Adjoint}, i, j)
    return (expose(parent(E))[j, i])'
end

function Base.any(f, E::Exposed{<:CuArray, <:NDTensors.Tensor})
    return any(f, NDTensors.data(unexpose(E)))
end

function Base.print_array(io::IO, E::Exposed{<:CuArray})
    return Base.print_array(io, expose(NDTensors.cpu(E)))
end
