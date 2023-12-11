default_label_name_type() = String7
default_category() = U(1)
default_values_type() = Tuple{Int,Int}

# TODO Temporary definition:
convert_values(::Type{Tuple{Int,Int}},t::Tuple) = t
convert_values(::Type{Tuple{Int,Int}},i::Int) = (i,0)

"""
Label of a specific sector or irrep of a category or group.
A Label also carries a name such as "J" or "Sz" giving the 
meaning of the symmetry in an application.
"""
struct Label{Name,Values,Category}
  name::Name
  values::Values
  category::Category

  global function _Label(name,values,category)
    return new{typeof(name),typeof(values),typeof(category)}(name,values,category)
  end
end

function Label(name="", values=0, category=U(1);
               name_type=default_label_name_type(),
               values_type=default_values_type(),
               default_category=default_category())
  return _Label(convert(name_type,name),convert_values(values_type,values),category)
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
    Label(name(s1), v, category(s1)) for v in fusion_rule(category(s1), values(s1), values(s2))
  ]
end
Base.:(*)(s1::Label, s2::Label) = ⊗(s1, s2)

function Base.show(io::IO, l::Label)
  return print(io, "Label(\"", name(l), "\",", values(l), ",", category(l), ")")
end
