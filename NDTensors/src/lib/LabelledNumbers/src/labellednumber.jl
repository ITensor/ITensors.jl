struct LabelledNumber{Value<:Number,Label} <: Number
  value::Value
  label::Label
end
LabelledStyle(::Type{<:LabelledNumber}) = IsLabelled()
label(lobject::LabelledNumber) = lobject.label
# TODO: Use `TypeParameterAccessors`.
label_type(::Type{<:LabelledNumber{<:Any,Label}}) where {Label} = Label
labelled(object::Number, label) = LabelledNumber(object, label)
unlabel(lobject::LabelledNumber) = lobject.value
unlabel_type(::Type{<:LabelledNumber{Value}}) where {Value} = Value

# TODO: Define `labelled_convert`.
Base.convert(type::Type{<:Number}, x::LabelledNumber) = type(unlabel(x))
# TODO: Define `labelled_promote_type`.
function Base.promote_type(type1::Type{T}, type2::Type{T}) where {T<:LabelledNumber}
  return promote_type(unlabel_type(type1), unlabel_type(type2))
end
function Base.promote_rule(type1::Type{<:LabelledNumber}, type2::Type{<:LabelledNumber})
  return promote_type(unlabel_type(type1), unlabel_type(type2))
end
function Base.promote_rule(type1::Type{<:LabelledNumber}, type2::Type{<:Number})
  return promote_type(unlabel_type(type1), type2)
end

Base.:(==)(x::LabelledNumber, y::LabelledNumber) = labelled_isequal(x, y)
Base.:<(x::LabelledNumber, y::LabelledNumber) = labelled_isless(x < y)
# TODO: Define `labelled_colon`.
(::Base.Colon)(start::LabelledNumber, stop::LabelledNumber) = unlabel(start):unlabel(stop)
Base.zero(lobject::LabelledNumber) = labelled_zero(lobject)
Base.one(lobject::LabelledNumber) = labelled_one(lobject)
Base.one(type::Type{<:LabelledNumber}) = labelled_one(type)
Base.oneunit(lobject::LabelledNumber) = labelled_oneunit(lobject)
Base.oneunit(type::Type{<:LabelledNumber}) = error("Not implemented.")

Base.:*(x::LabelledNumber, y::LabelledNumber) = labelled_mul(x, y)
Base.:*(x::LabelledNumber, y::Number) = labelled_mul(x, y)
Base.:*(x::Number, y::LabelledNumber) = labelled_mul(x, y)

Base.:/(x::LabelledNumber, y::Number) = labelled_division(x, y)
Base.div(x::LabelledNumber, y::Number) = labelled_div(x, y)

# TODO: This is only needed for older Julia versions, like Julia 1.6.
# Delete once we drop support for older Julia versions.
# TODO: Define in terms of a generic `labelled_minus` function.
# TODO: Define in terms of `set_value`?
Base.:-(x::LabelledNumber) = labelled_minus(x)

# TODO: This is only needed for older Julia versions, like Julia 1.6.
# Delete once we drop support for older Julia versions.
Base.hash(x::LabelledNumber, h::UInt64) = labelled_hash(x, h)
