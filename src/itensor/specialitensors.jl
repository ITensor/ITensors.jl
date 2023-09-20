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
  A = hasqns(A) ? dropzeros(A) : A
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

## from QN ITensors

"""
    delta([::Type{ElT} = Float64, ][flux::QN = QN(), ]is)
    delta([::Type{ElT} = Float64, ][flux::QN = QN(), ]is::Index...)

Make an ITensor with storage type `NDTensors.DiagBlockSparse` with uniform
elements `one(ElT)`. The ITensor only has diagonal blocks consistent with the
specified `flux`.

If the element type is not specified, it defaults to `Float64`. If theflux is
not specified, it defaults to `QN()`.
"""
function delta(::Type{ElT}, flux::QN, inds::Indices) where {ElT<:Number}
  is = Tuple(inds)
  blocks = nzdiagblocks(flux, is)
  T = DiagBlockSparseTensor(one(ElT), blocks, is)
  return itensor(T)
end

function delta(::Type{ElT}, flux::QN, is...) where {ElT<:Number}
  return delta(ElT, flux, indices(is...))
end

delta(flux::QN, inds::Indices) = delta(Float64, flux, is)

delta(flux::QN, is...) = delta(Float64, flux, indices(is...))

function delta(::Type{ElT}, inds::QNIndices) where {ElT<:Number}
  return delta(ElT, QN(), inds)
end

delta(inds::QNIndices) = delta(Float64, QN(), inds)
