"""
    delta([::Type{ElT} = Float64, ]inds)
    delta([::Type{ElT} = Float64, ]inds::Index...)

Make a uniform diagonal ITensor with all diagonal elements
`one(ElT)`. Only a single diagonal element is stored.

This function has an alias `δ`.
"""
function delta(eltype::Type{<:Number}, is::Indices)
  return itensor(Diag(one(eltype)), is)
end

function delta(eltype::Type{<:Number}, is...)
  return delta(eltype, indices(is...))
end

delta(is...) = delta(Float64, is...)

const δ = delta

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
function onehot(datatype::Type{<:AbstractArray}, ivs::Pair{<:Index}...)
  A = ITensor(eltype(datatype), ind.(ivs)...)
  A[val.(ivs)...] = one(eltype(datatype))
  return Adapt.adapt(datatype, A)
end

function onehot(eltype::Type{<:Number}, ivs::Pair{<:Index}...)
  return onehot(NDTensors.default_datatype(eltype), ivs...)
end
function onehot(eltype::Type{<:Number}, ivs::Vector{<:Pair{<:Index}})
  return onehot(NDTensors.default_datatype(eltype), ivs...)
end
function setelt(eltype::Type{<:Number}, ivs::Pair{<:Index}...)
  return onehot(NDTensors.default_datatype(eltype), ivs...)
end

function onehot(ivs::Pair{<:Index}...)
  return onehot(NDTensors.default_datatype(NDTensors.default_eltype()), ivs...)
end
onehot(ivs::Vector{<:Pair{<:Index}}) = onehot(ivs...)
setelt(ivs::Pair{<:Index}...) = onehot(ivs...)

### informally defined Scalar ITensors

# For now, it's not well defined to construct an ITensor without indices
# from a non-zero dimensional Array.
function ITensor(
  as::AliasStyle, ElT::Type{<:Number}, A::AbstractArray{<:Number}; kwargs...
)
  if length(A) > 1
    error(
      "Trying to create an ITensor without any indices from Array $A of dimensions $(size(A)). Cannot construct an ITensor from an Array with more than one element without any indices.",
    )
  end
  return ITensor(as, ElT, A, Index(1); kwargs...)
end

function ITensor(eltype::Type{<:Number}, A::AbstractArray{<:Number}; kwargs...)
  return ITensor(NeverAlias(), eltype, A; kwargs...)
end

function ITensor(A::AbstractArray; kwargs...)
  return ITensor(NeverAlias(), eltype(A), A; kwargs...)
end
