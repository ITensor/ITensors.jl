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

# TODO: Is this a good definition?
Base.unitrange(a::LabelledUnitRange) = a

for f in [:first, :getindex, :last, :length, :step]
  @eval Base.$f(a::LabelledUnitRange, args...) = labelled($f(unlabel(a), args...), label(a))
end

labelled_getindex(a, index) = labelled(unlabel(a)[index], label(a))

# This is required in Julia 1.11 and above since
# the generic `axes(a::AbstractRange)` definition was removed
# and replace with a generic `axes(a)` definition that
# is written in terms of `Base.unchecked_oneto`, i.e.:
# ```julia
# map(Base.unchecked_oneto, size(A))
# ```
# which returns a `Base.OneTo` instead of a `LabelledUnitRange`.
Base.axes(a::LabelledUnitRange) = Base.oneto.(size(a))

# TODO: Delete this definition, this should output a `Base.OneTo`.
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

function Base.show(io::IO, ::MIME"text/plain", a::LabelledUnitRange)
  println(io, typeof(a))
  return print(io, label(a), " => ", unlabel(a))
end

function Base.show(io::IO, a::LabelledUnitRange)
  return print(io, nameof(typeof(a)), " ", label(a), " => ", unlabel(a))
end
