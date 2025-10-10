module ITensorsSiteTypesExt
using ..ITensors: ITensors, Index, LastVal, dag, prime, val
using NDTensors: NDTensors, dim, sim
using ..SiteTypes: SiteTypes
SiteTypes.val(iv::Pair{<:Index}) = val(iv.first, iv.second)
SiteTypes.val(i::Index, l::LastVal) = l.f(dim(i))
# TODO:
# Implement a macro with a general definition:
# f(iv::Pair{<:Index}, args...) = (f(ind(iv), args...) => val(iv))
ITensors.prime(iv::Pair{<:Index}, inc::Integer = 1) = (prime(ind(iv), inc) => val(iv))
NDTensors.sim(iv::Pair{<:Index}, args...) = (sim(ind(iv), args...) => val(iv))
ITensors.dag(iv::Pair{<:Index}) = (dag(ind(iv)) => val(iv))
Base.adjoint(iv::Pair{<:Index}) = (prime(ind(iv)) => val(iv))

using ..ITensors: ITensors, Indices
function ITensors._vals(is::Indices, I::String...)
    return val.(is, I)
end

using Adapt: Adapt
using ..ITensors: ITensors, Index, ITensor, ind, inds
using NDTensors: NDTensors, Tensor
using ..SiteTypes: val
Base.@propagate_inbounds @inline function ITensors._getindex(
        T::Tensor, ivs::Vararg{Any, N}
    ) where {N}
    # Tried ind.(ivs), val.(ivs) but it is slower
    p = NDTensors.getperm(inds(T), ntuple(n -> ind(@inbounds ivs[n]), Val(N)))
    fac = NDTensors.permfactor(p, ivs...) #<fermions> possible sign
    return fac * ITensors._getindex(
        T, NDTensors.permute(ntuple(n -> val(@inbounds ivs[n]), Val(N)), p)...
    )
end
Base.@propagate_inbounds @inline function ITensors._setindex!!(
        T::Tensor, x::Number, ivs::Vararg{Any, N}
    ) where {N}
    # Would be nice to split off the functions for extracting the `ind` and `val` as Tuples,
    # but it was slower.
    p = NDTensors.getperm(inds(T), ntuple(n -> ind(@inbounds ivs[n]), Val(N)))
    fac = NDTensors.permfactor(p, ivs...) #<fermions> possible sign
    return ITensors._setindex!!(
        T, fac * x, NDTensors.permute(ntuple(n -> val(@inbounds ivs[n]), Val(N)), p)...
    )
end
Base.@propagate_inbounds @inline function Base.setindex!(
        T::ITensor, x::Number, I1::Pair{<:Index, String}, I::Pair{<:Index, String}...
    )
    Iv = map(i -> i.first => val(i.first, i.second), (I1, I...))
    return setindex!(T, x, Iv...)
end
"""
    onehot(ivs...)
    setelt(ivs...)
    onehot(::Type, ivs...)
    setelt(::Type, ivs...)

Create an ITensor with all zeros except the specified value,
which is set to 1.

# Examples
```julia
i = Index(2,"i")
A = onehot(i=>2)
# A[i=>2] == 1, all other elements zero

# Specify the element type
A = onehot(Float32, i=>2)

j = Index(3,"j")
B = onehot(i=>1,j=>3)
# B[i=>1,j=>3] == 1, all other element zero
```
"""
function ITensors.onehot(datatype::Type{<:AbstractArray}, ivs::Pair{<:Index}...)
    A = ITensor(eltype(datatype), ind.(ivs)...)
    A[val.(ivs)...] = one(eltype(datatype))
    return Adapt.adapt(datatype, A)
end
end
