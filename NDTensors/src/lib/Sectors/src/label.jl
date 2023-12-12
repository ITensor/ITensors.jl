default_label_name_type() = String7
default_category() = U(1)
default_values_type() = Tuple{Half{Int},Half{Int}}

# TODO Temporary definition:
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
struct Label{Name,Values,Category}
  name::Name
  values::Values
  category::Category

  global function _Label(name, values, category)
    return new{typeof(name),typeof(values),typeof(category)}(name, values, category)
  end
end

# TODO: cleanup default category
function Label(
  name::AbstractString="",
  values=0,
  category::Union{Category,Nothing}=nothing;
  name_type=default_label_name_type(),
  values_type=default_values_type(),
  default_category=default_category(),
)
  isnothing(category) && (category = default_category)
  return _Label(
    convert(name_type, name), convert_values(values_type, values, category), category
  )
end

# TODO: fix or rule out ambiguity error
# when values is a string
function Label(values, category::Union{Category,Nothing}=nothing; kws...)
  return Label("", values, category; kws...)
end

#function Label(name::AbstractString, val::AbstractString, cat=U(1))
#  return Label(name, str_to_val(cat, val), 0, cat)
#end
#Label(name::AbstractString, vals::Tuple, cat=U(1)) = Label(name, vals[1], vals[2], cat)

#const Label7 = Label{String7,Tuple{Int,Int},Category7}

name(s::Label) = s.name
values(s::Label) = s.values
category(s::Label) = s.category

# TODO: still needed?
# TODO: add tests for this
function val_to_str(s::Label)
  if nvals(category(s)) == 1
    return val_to_str(category(s), values(s))
  end
  return val_to_str(category(s), values(s))
end

function ⊗(s1::Label, s2::Label)
  name(s1) == name(s2) ||
    error("Cannot fuse sector labels with different names \"$(name(s1))\", \"$(name(s2))\"")
  category(s1) == category(s2) ||
    error("Labels with matching name \"$(name(s1))\" cannot have different category types")
  return [
    Label(name(s1), v, category(s1)) for
    v in fusion_rule(category(s1), values(s1), values(s2))
  ]
end
Base.:(*)(s1::Label, s2::Label) = ⊗(s1, s2)

function Base.show(io::IO, l::Label)
  return print(io, "Label(\"", name(l), "\",", values(l), ",", category(l), ")")
end

istrivial(l::Label) = istrivial(category(l), values(l))
