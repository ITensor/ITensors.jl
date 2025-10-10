using NDTensors.Expose: Exposed, expose, parent, unexpose
using AMDGPU: ROCArray

function Base.permutedims!(
        Edest::Exposed{<:ROCArray, <:Base.ReshapedArray}, Esrc::Exposed{<:ROCArray}, perm
    )
    Aperm = permutedims(Esrc, perm)
    copyto!(expose(parent(Edest)), expose(Aperm))
    return unexpose(Edest)
end

# There is an issue in AMDGPU where if Edest is a reshaped{<:Adjoint}
# .= can fail. So instead force Esrc into the shape of parent(Edest)
function Base.permutedims!(
        Edest::Exposed{<:ROCArray, <:Base.ReshapedArray{<:Any, <:Any, <:Adjoint}},
        Esrc::Exposed{<:ROCArray},
        perm,
        f,
    )
    Aperm = reshape(permutedims(Esrc, perm), size(parent(Edest)))
    parent(Edest) .= f.(parent(Edest), Aperm)
    return unexpose(Edest)
end
