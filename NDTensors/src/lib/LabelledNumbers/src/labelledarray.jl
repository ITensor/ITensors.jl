struct LabelledArray{T,N,Value<:AbstractArray{T,N},Label} <:
       AbstractArray{LabelledInteger{T,Label},N}
  value::Value
  label::Label
end
LabelledStyle(::Type{<:LabelledArray}) = IsLabelled()
label(lobject::LabelledArray) = lobject.label
# TODO: Use `TypeParameterAccessors`.
label_type(::Type{<:LabelledArray{<:Any,Label}}) where {Label} = Label
labelled(object::AbstractArray, label) = LabelledArray(object, label)
unlabel(lobject::LabelledArray) = lobject.value
unlabel_type(::Type{<:LabelledArray{Value}}) where {Value} = Value

for f in [:axes]
  @eval Base.$f(a::LabelledArray, args...) = $f(unlabel(a), args...)
end

for f in [:first, :getindex, :last, :length]
  @eval Base.$f(a::LabelledArray, args...) = labelled($f(unlabel(a), args...), label(a))
end
