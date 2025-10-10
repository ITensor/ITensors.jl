using JLArrays: JLArray
using NDTensors.Expose: Exposed, expose, unexpose
using LinearAlgebra: Adjoint

# Same definition as `CuArray`.
function Base.copy(src::Exposed{<:JLArray, <:Base.ReshapedArray})
    return reshape(copy(parent(src)), size(unexpose(src)))
end

function Base.copy(
        src::Exposed{
            <:JLArray, <:SubArray{<:Any, <:Any, <:Base.ReshapedArray{<:Any, <:Any, <:Adjoint}},
        },
    )
    return copy(@view copy(expose(parent(src)))[parentindices(unexpose(src))...])
end

# Catches a bug in `copyto!` in CUDA backend.
function Base.copyto!(dest::Exposed{<:JLArray}, src::Exposed{<:JLArray, <:SubArray})
    copyto!(dest, expose(copy(src)))
    return unexpose(dest)
end

# Catches a bug in `copyto!` in JLArray backend.
function Base.copyto!(
        dest::Exposed{<:JLArray}, src::Exposed{<:JLArray, <:Base.ReshapedArray}
    )
    copyto!(dest, expose(parent(src)))
    return unexpose(dest)
end
