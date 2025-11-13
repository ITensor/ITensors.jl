using NDTensors.Expose: Exposed, expose, parent, unexpose
using LinearAlgebra: LinearAlgebra, Adjoint
using AMDGPU: ROCArray

# Same definition as `MtlArray`.
function Base.copy(src::Exposed{<:ROCArray, <:Base.ReshapedArray})
    return reshape(copy(parent(src)), size(unexpose(src)))
end

function Base.copy(
        src::Exposed{
            <:ROCArray, <:SubArray{<:Any, <:Any, <:Base.ReshapedArray{<:Any, <:Any, <:Adjoint}},
        },
    )
    return copy(@view copy(expose(parent(src)))[parentindices(unexpose(src))...])
end

function Base.copyto!(dest::Exposed{<:ROCArray}, src::Exposed{<:ROCArray, <:SubArray})
    copyto!(dest, expose(copy(src)))
    return unexpose(dest)
end

function Base.copyto!(
        dest::Exposed{<:ROCArray}, src::Exposed{<:ROCArray, <:Base.ReshapedArray}
    )
    copyto!(dest, expose(parent(src)))
    return unexpose(dest)
end

function Base.copyto!(
        dest::Exposed{<:ROCArray}, src::Exposed{<:ROCArray, <:LinearAlgebra.Transpose}
    )
    copyto!(expose(transpose(dest)), expose(parent(src)))
    return unexpose(dest)
end
