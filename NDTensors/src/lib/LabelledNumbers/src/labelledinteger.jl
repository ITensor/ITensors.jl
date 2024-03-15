struct LabelledInteger{Value<:Integer,Label} <: Integer
  value::Value
  label::Label
end
LabelledStyle(::Type{<:LabelledInteger}) = IsLabelled()
label(lobject::LabelledInteger) = lobject.label
# TODO: Use `TypeParameterAccessors`.
label_type(::Type{<:LabelledInteger{<:Any,Label}}) where {Label} = Label
labelled(object::Integer, label) = LabelledInteger(object, label)
unlabel(lobject::LabelledInteger) = lobject.value
unlabel_type(::Type{<:LabelledInteger{Value}}) where {Value} = Value

# TODO: Define `labelled_convert`.
Base.convert(type::Type{<:Number}, x::LabelledInteger) = type(unlabel(x))
# TODO: This is only needed for older Julia versions, like Julia 1.6.
# Delete once we drop support for older Julia versions.
function Base.convert(type::Type{<:LabelledInteger}, x::LabelledInteger)
  return type(unlabel(x), label(x))
end
# TODO: Define `labelled_promote_type`.
function Base.promote_type(type1::Type{T}, type2::Type{T}) where {T<:LabelledInteger}
  return promote_type(unlabel_type(type1), unlabel_type(type2))
end
function Base.promote_rule(type1::Type{<:LabelledInteger}, type2::Type{<:LabelledInteger})
  return promote_type(unlabel_type(type1), unlabel_type(type2))
end
function Base.promote_rule(type1::Type{<:LabelledInteger}, type2::Type{<:Number})
  return promote_type(unlabel_type(type1), type2)
end

Base.:(==)(x::LabelledInteger, y::LabelledInteger) = labelled_isequal(x, y)
Base.:<(x::LabelledInteger, y::LabelledInteger) = labelled_isless(x, y)
# TODO: Define `labelled_colon`.
(::Base.Colon)(start::LabelledInteger, stop::LabelledInteger) = unlabel(start):unlabel(stop)
# TODO: Define `labelled_one`.
Base.one(lobject::LabelledInteger) = one(unlabel(lobject))
# TODO: Define `labelled_oneunit`.
Base.oneunit(lobject::LabelledInteger) = labelled(one(lobject), label(lobject))

Base.:*(x::LabelledInteger, y::LabelledInteger) = labelled_mul(x, y)
Base.:*(x::LabelledInteger, y::Number) = labelled_mul(x, y)
Base.:*(x::Number, y::LabelledInteger) = labelled_mul(x, y)
# Fix ambiguity issue with `Base` `Integer`.
Base.:*(x::LabelledInteger, y::Integer) = labelled_mul(x, y)
# Fix ambiguity issue with `Base` `Integer`.
Base.:*(x::Integer, y::LabelledInteger) = labelled_mul(x, y)

Base.:/(x::LabelledInteger, y::Number) = labelled_division(x, y)
Base.div(x::LabelledInteger, y::Number) = labelled_div(x, y)
