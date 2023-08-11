#
# Diag ITensor constructors
#

"""
    diagITensor([ElT::Type, ]v::Vector, inds...)
    diagitensor([ElT::Type, ]v::Vector, inds...)

Make a sparse ITensor with non-zero elements only along the diagonal.
In general, the diagonal elements will be those stored in `v` and
the ITensor will have element type `eltype(v)`, unless specified explicitly
by `ElT`. The storage will have `NDTensors.Diag` type.

NOTE: In the case when `eltype(v) isa Union{Int, Complex{Int}}`, and the element type `elt` 
is not specified explicitly, v will be converted to `float(v)`. Note that this behavior is subject to change
in the future.

The version `diagITensor` will never output an ITensor whose storage data
is an alias of the input vector data.

The version `diagitensor` might output an ITensor whose storage data
is an alias of the input vector data in order to minimize operations.
"""
function diagITensor(
  as::AliasStyle, elt::Type{<:Number}, v::AbstractVector{<:Number}, is::Indices
)
  length(v) ≠ mindim(is) && error(
    "Length of vector for diagonal must equal minimum of the dimension of the input indices",
  )
  data = set_eltype(typeof(v), elt)(as, v)
  return itensor(Diag(data), is)
end

function diagITensor(
  as::AliasStyle, elt::Type{<:Number}, v::AbstractVector{<:Number}, is...
)
  return diagITensor(as, elt, v, indices(is...))
end

function diagITensor(as::AliasStyle, v::AbstractVector, is...)
  return diagITensor(as, eltype(v), v, is...)
end

function diagITensor(as::AliasStyle, v::AbstractVector{<:RealOrComplex{Int}}, is...)
  return diagITensor(as, float(eltype(v)), v, is...)
end

diagITensor(v::AbstractVector{<:Number}, is...) = diagITensor(NeverAlias(), v, is...)
function diagITensor(elt::Type{<:Number}, v::AbstractVector{<:Number}, is...)
  return diagITensor(NeverAlias(), elt, v, is...)
end

diagitensor(args...; kwargs...) = diagITensor(AllowAlias(), args...; kwargs...)

# XXX TODO: explain conversion from Int
# XXX TODO: proper conversion
"""
    diagITensor([elt::Type, ]x::Number, inds...)
    diagitensor([elt::Type, ]x::Number, inds...)

Make a sparse ITensor with non-zero elements only along the diagonal.
In general, the diagonal elements will be set to the value `x` and
the ITensor will have element type `eltype(x)`, unless specified explicitly
by `ElT`. The storage will have `NDTensors.Diag` type.

In the case when `x isa Union{Int, Complex{Int}}`, by default it will
be converted to `float(x)`. Note that this behavior is subject to change
in the future.
"""
function diagITensor(::AliasStyle, elt::Type{<:Number}, x::Number, is::Indices)
  return diagITensor(AllowAlias(), elt, fill(x, mindim(is)), is)
end

function diagITensor(as::AliasStyle, elt::Type{<:Number}, x::Number, is...)
  return diagITensor(as, elt, x, indices(is...))
end

function diagITensor(as::AliasStyle, x::Number, is...)
  return diagITensor(as, typeof(x), x, is...)
end

function diagITensor(as::AliasStyle, x::RealOrComplex{Int}, is...)
  return diagITensor(as, float(typeof(x)), x, is...)
end

function diagITensor(elt::Type{<:Number}, x::Number, is...)
  return diagITensor(NeverAlias(), elt, x, is...)
end

"""
    diagITensor([::Type{ElT} = Float64, ]inds)
    diagITensor([::Type{ElT} = Float64, ]inds::Index...)

Make a sparse ITensor of element type `ElT` with only elements
along the diagonal stored. Defaults to having `zero(T)` along
the diagonal.

The storage will have `NDTensors.Diag` type.
"""
function diagITensor(::Type{ElT}, is::Indices) where {ElT<:Number}
  return diagITensor(NeverAlias(), ElT, 0, is)
end

diagITensor(::Type{ElT}, is...) where {ElT<:Number} = diagITensor(ElT, indices(is))

diagITensor(is::Indices) = diagITensor(NDTensors.default_eltype(), is)
diagITensor(is...) = diagITensor(indices(is...))

diagITensor(x::Number, is...) = diagITensor(NeverAlias(), x, is...)
