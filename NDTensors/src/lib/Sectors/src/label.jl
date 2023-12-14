default_label_name_type() = String7
default_values_type() = Tuple{Half{Int},Half{Int}}

#
# TODO Temporary definition:
#
function convert_values(::Type{default_values_type()}, t::NTuple{N,T}, cat) where {N,T}
  return ntuple(j -> convert(Half{Int}, t[j]), Val(N))
end
function convert_values(::Type{default_values_type()}, v::Number, cat)
  return (convert(Half{Int}, v), zero(Half{Int}))
end
function convert_values(T::Type{default_values_type()}, s::AbstractString, cat)
  return convert_values(T, str_to_val(cat, s), cat)
end

"""
Label of a specific sector or irrep of a category or group.
A Label also carries a name such as "J" or "Sz" giving the 
meaning of the symmetry in an application.
"""
struct Label{Name,Category,Values} <: AbstractLabel
  name::Name
  category::Category
  values::Values

  global function _Label(name, category, values)
    return new{typeof(name),typeof(category),typeof(values)}(name, category, values)
  end
end

function Label(
  name::AbstractString,
  category::Category,
  values=0;
  name_type=default_label_name_type(),
  values_type=default_values_type(),
)
  return _Label(
    convert(name_type, name), category, convert_values(values_type, values, category)
  )
end

Label(category::Category, values; kws...) = Label("", category, values; kws...)

name(s::Label) = s.name
category(s::Label) = s.category
Base.values(s::Label) = s.values

function ⊗(s1::Label, s2::Label)
  name(s1) == name(s2) ||
    error("Cannot fuse sector labels with different names \"$(name(s1))\", \"$(name(s2))\"")
  category(s1) == category(s2) ||
    error("Labels with matching name \"$(name(s1))\" cannot have different category types")
  return [
    Label(name(s1), category(s1), v) for
    v in fusion_rule(category(s1), values(s1), values(s2))
  ]
end

istrivial(l::Label) = istrivial(category(l), values(l))

val_to_str(l::Label) = val_to_str(category(l), values(l))

function Base.show(io::IO, l::Label)
  print(io, "Label(")
  (length(name(l)) > 0) && print(io, "\"", name(l), "\",")
  return print(io, category(l), ",", val_to_str(l), ")")
end
