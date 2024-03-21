struct LabelledUnitRange{T,Value<:AbstractUnitRange{T},Label} <:
       AbstractUnitRange{LabelledInteger{T,Label}}
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

# Used by `CartesianIndices` constructor.
# TODO: Maybe reconsider this definition? Also, this should preserve
# the label if possible, currently it drops the label.
function Base.AbstractUnitRange{T}(a::LabelledUnitRange) where {T}
  return AbstractUnitRange{T}(unlabel(a))
end

for f in [:first, :getindex, :last, :length, :step]
  @eval Base.$f(a::LabelledUnitRange, args...) = labelled($f(unlabel(a), args...), label(a))
end

labelled_getindex(a, index) = labelled(unlabel(a)[index], label(a))

Base.OneTo(stop::LabelledInteger) = labelled(Base.OneTo(unlabel(stop)), label(stop))

# Fix ambiguity error with `AbstractRange` definition in `Base`.
Base.getindex(a::LabelledUnitRange, index::Integer) = labelled_getindex(a, index)
# Fix ambiguity error with `AbstractRange` definition in `Base`.
function Base.getindex(a::LabelledUnitRange, indices::AbstractUnitRange{<:Integer})
  return labelled_getindex(a, indices)
end

function Base.iterate(a::LabelledUnitRange, i)
  i == last(a) && return nothing
  next = convert(eltype(a), labelled(i + step(a), label(a)))
  return (next, next)
end
