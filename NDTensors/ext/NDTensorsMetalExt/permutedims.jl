using Metal: MtlArray
using GPUArraysCore: @allowscalar
using NDTensors.Expose: Exposed, expose, unexpose
## Theres an issue in metal that `ReshapedArray' wrapped arrays cannot be permuted using
## permutedims (failing in that Metal uses scalar indexing)
## These functions are to address the problem in different instances of permutedims
function Base.permutedims(E::Exposed{<:MtlArray, <:Base.ReshapedArray}, perm)
    A = copy(E)
    return permutedims(A, perm)
end

function Base.permutedims!(
        Edest::Exposed{<:MtlArray, <:Base.ReshapedArray}, Esrc::Exposed{<:MtlArray}, perm
    )
    Aperm = permutedims(Esrc, perm)
    copyto!(expose(parent(Edest)), expose(Aperm))
    return unexpose(Edest)
end

function Base.permutedims!(
        Edest::Exposed{<:MtlArray}, Esrc::Exposed{<:MtlArray, <:Base.ReshapedArray}, perm
    )
    Aperm = permutedims(Esrc, perm)
    copyto!(Edest, expose(Aperm))
    return unexpose(Edest)
end

## To get around the Metal issue here we copy and permute Esrc,
## then we reshape Esrc to the size of Edest's parent
## and broadcast into the parent.
function Base.permutedims!(
        Edest::Exposed{<:MtlArray, <:Base.ReshapedArray},
        Esrc::Exposed{<:MtlArray, <:Base.ReshapedArray},
        perm,
        f,
    )
    Aperm = reshape(permutedims(Esrc, perm), size(parent(Edest)))
    parent(Edest) .= f.(parent(Edest), Aperm)
    return unexpose(Edest)
end
