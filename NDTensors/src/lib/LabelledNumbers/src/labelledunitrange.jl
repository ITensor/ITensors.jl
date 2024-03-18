struct LabelledUnitRange{T,Value<:AbstractUnitRange{T},Label} <: AbstractUnitRange{T}
  value::Value
  label::Label
end
LabelledStyle(::Type{<:LabelledUnitRange}) = IsLabelled()
label(lobject::LabelledUnitRange) = lobject.label
# TODO: Use `TypeParameterAccessors`.
label_type(::Type{<:LabelledUnitRange{<:Any,Label}}) where {Label} = Label
labelled(object::AbstractUnitRange, label) = LabelledUnitRange(object, label)
unlabel(lobject::LabelledUnitRange) = lobject.value
unlabel_type(::Type{<:LabelledUnitRange{Value}}) where {Value} = Value

for f in [:first, :getindex, :last, :length]
  @eval Base.$f(a::LabelledUnitRange, args...) = labelled($f(unlabel(a), args...), label(a))
end

labelled_getindex(a, index) = labelled(unlabel(a)[index], label(a))

# Fix ambiguity error with `AbstractRange` definition in `Base`.
Base.getindex(a::LabelledUnitRange, index::Integer) = labelled_getindex(a, index)
# Fix ambiguity error with `AbstractRange` definition in `Base`.
function Base.getindex(a::LabelledUnitRange, indices::AbstractUnitRange{<:Integer})
  return labelled_getindex(a, indices)
end
