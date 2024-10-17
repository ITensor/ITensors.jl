struct LabelledInteger{Value<:Integer,Label} <: Integer
  value::Value
  label::Label
end
LabelledStyle(::Type{<:LabelledInteger}) = IsLabelled()
# TODO: Define `set_value` and `set_label`?
label(lobject::LabelledInteger) = lobject.label
# TODO: Use `TypeParameterAccessors`.
label_type(::Type{<:LabelledInteger{<:Any,Label}}) where {Label} = Label
labelled(object::Integer, label) = LabelledInteger(object, label)
unlabel(lobject::LabelledInteger) = lobject.value
unlabel_type(::Type{<:LabelledInteger{Value}}) where {Value} = Value

# When using as shapes of arrays.
# TODO: Preserve the label? For example:
# labelled(Base.to_shape(unlabel(x)), label(x))
Base.to_shape(x::LabelledInteger) = Base.to_shape(unlabel(x))

# TODO: Define `labelled_convert`.
Base.convert(type::Type{<:Number}, x::LabelledInteger) = type(unlabel(x))
# TODO: This is only needed for older Julia versions, like Julia 1.6.
# Delete once we drop support for older Julia versions.
function Base.convert(type::Type{<:LabelledInteger}, x::LabelledInteger)
  return type(unlabel(x), label(x))
end

# Used by `Base.hash(::Integer)`.
# TODO: Define `labelled_trailing_zeros` to be used by other
# labelled number types.
Base.trailing_zeros(x::LabelledInteger) = trailing_zeros(unlabel(x))

# Used by `Base.hash(::Integer)`.
# TODO: Define `labelled_right_bit_shift` to be used by other
# labelled number types.
Base.:>>(x::LabelledInteger, y::Int) = >>(unlabel(x), y)

Base.:(==)(x::LabelledInteger, y::LabelledInteger) = labelled_isequal(x, y)
Base.:(==)(x::LabelledInteger, y::Number) = labelled_isequal(x, y)
Base.:(==)(x::Number, y::LabelledInteger) = labelled_isequal(x, y)
Base.:<(x::LabelledInteger, y::LabelledInteger) = labelled_isless(x, y)
# This is only needed on older versions of Julia, like Julia 1.6.
# TODO: Delete once we drop support for Julia 1.6.
function Base.:<=(x::LabelledInteger, y::LabelledInteger)
  return labelled_isless(x, y) || labelled_isequal(x, y)
end
# TODO: Define `labelled_colon`.
(::Base.Colon)(start::LabelledInteger, stop::LabelledInteger) = unlabel(start):unlabel(stop)
Base.zero(lobject::LabelledInteger) = labelled_zero(lobject)
Base.one(lobject::LabelledInteger) = labelled_one(lobject)
Base.one(type::Type{<:LabelledInteger}) = labelled_one(type)
Base.oneunit(lobject::LabelledInteger) = labelled_oneunit(lobject)
Base.oneunit(type::Type{<:LabelledInteger}) = oneunit(unlabel_type(type))
Base.zero(type::Type{<:LabelledInteger}) = zero(unlabel_type(type))

Base.Int(x::LabelledInteger) = Int(unlabel(x))

Base.:+(x::LabelledInteger, y::LabelledInteger) = labelled_add(x, y)
Base.:+(x::LabelledInteger, y::Number) = labelled_add(x, y)
Base.:+(x::Number, y::LabelledInteger) = labelled_add(x, y)
# Fix ambiguity error with `+(::Integer, ::Integer)`.
Base.:+(x::LabelledInteger, y::Integer) = labelled_add(x, y)
Base.:+(x::Integer, y::LabelledInteger) = labelled_add(x, y)

Base.:-(x::LabelledInteger, y::LabelledInteger) = labelled_minus(x, y)
Base.:-(x::LabelledInteger, y::Number) = labelled_minus(x, y)
Base.:-(x::Number, y::LabelledInteger) = labelled_minus(x, y)
# Fix ambiguity error with `-(::Integer, ::Integer)`.
Base.:-(x::LabelledInteger, y::Integer) = labelled_minus(x, y)
Base.:-(x::Integer, y::LabelledInteger) = labelled_minus(x, y)

function Base.sub_with_overflow(x::LabelledInteger, y::LabelledInteger)
  return labelled_binary_op(Base.sub_with_overflow, x, y)
end

Base.:*(x::LabelledInteger, y::LabelledInteger) = labelled_mul(x, y)
Base.:*(x::LabelledInteger, y::Number) = labelled_mul(x, y)
Base.:*(x::Number, y::LabelledInteger) = labelled_mul(x, y)
# Fix ambiguity issue with `Base` `Integer`.
Base.:*(x::LabelledInteger, y::Integer) = labelled_mul(x, y)
# Fix ambiguity issue with `Base` `Integer`.
Base.:*(x::Integer, y::LabelledInteger) = labelled_mul(x, y)

Base.:/(x::LabelledInteger, y::Number) = labelled_division(x, y)
Base.div(x::LabelledInteger, y::Number) = labelled_div(x, y)

# TODO: This is only needed for older Julia versions, like Julia 1.6.
# Delete once we drop support for older Julia versions.
# TODO: Define in terms of a generic `labelled_minus` function.
# TODO: Define in terms of `set_value`?
Base.:-(x::LabelledInteger) = labelled_minus(x)

# TODO: This is only needed for older Julia versions, like Julia 1.6.
# Delete once we drop support for older Julia versions.
Base.hash(x::LabelledInteger, h::UInt64) = labelled_hash(x, h)

using Random: AbstractRNG, default_rng
default_eltype() = Float64
for f in [:rand, :randn]
  @eval begin
    function Base.$f(
      rng::AbstractRNG,
      elt::Type{<:Number},
      dims::Tuple{LabelledInteger,Vararg{LabelledInteger}},
    )
      return a = $f(rng, elt, unlabel.(dims))
    end
    function Base.$f(
      rng::AbstractRNG,
      elt::Type{<:Number},
      dim1::LabelledInteger,
      dims::Vararg{LabelledInteger},
    )
      return $f(rng, elt, (dim1, dims...))
    end
    Base.$f(elt::Type{<:Number}, dims::Tuple{LabelledInteger,Vararg{LabelledInteger}}) = $f(
      default_rng(), elt, dims
    )
    Base.$f(elt::Type{<:Number}, dim1::LabelledInteger, dims::Vararg{LabelledInteger}) = $f(
      elt, (dim1, dims...)
    )
    Base.$f(dims::Tuple{LabelledInteger,Vararg{LabelledInteger}}) = $f(
      default_eltype(), dims
    )
    Base.$f(dim1::LabelledInteger, dims::Vararg{LabelledInteger}) = $f((dim1, dims...))
  end
end
