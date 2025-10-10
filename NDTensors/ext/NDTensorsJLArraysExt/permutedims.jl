using JLArrays: JLArray
using LinearAlgebra: Adjoint
using NDTensors.Expose: Exposed, expose, unexpose

function Base.permutedims!(
        Edest::Exposed{<:JLArray, <:Base.ReshapedArray}, Esrc::Exposed{<:JLArray}, perm
    )
    Aperm = permutedims(Esrc, perm)
    copyto!(expose(parent(Edest)), expose(Aperm))
    return unexpose(Edest)
end

## Found an issue in CUDA where if Edest is a reshaped{<:Adjoint}
## .= can fail. So instead force Esrc into the shape of parent(Edest)
function Base.permutedims!(
        Edest::Exposed{<:JLArray, <:Base.ReshapedArray{<:Any, <:Any, <:Adjoint}},
        Esrc::Exposed{<:JLArray},
        perm,
        f,
    )
    Aperm = reshape(permutedims(Esrc, perm), size(parent(Edest)))
    parent(Edest) .= f.(parent(Edest), Aperm)
    return unexpose(Edest)
end
