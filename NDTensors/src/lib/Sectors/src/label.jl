"""
Label of a specific sector or irrep of a category or group.
A Label also carries a name such as "J" or "Sz" giving the 
meaning of the symmetry in an application.
"""
struct Label{Name,Values,Category}
  name::Name
  values::Values
  category::Category
end

Label() = Label("", 0, 0, Category())
Label(name::AbstractString, val::Number, cat=U(1)) = Label(name, val, 0, cat)
function Label(name::AbstractString, val::AbstractString, cat=U(1))
  return Label(name, str_to_val(cat, val), 0, cat)
end
Label(name::AbstractString, vals::Tuple, cat=U(1)) = Label(name, vals[1], vals[2], cat)

const Label7 = Label{String7,Tuple{Int,Int},Category7}


name(s::Label) = s.name
function val(s::Label)
  s.val2 != 0 && error("Second value of Label is non-zero = ($(s.val1),$(s.val2))")
  return s.val1
end
vals(s::Label) = (s.val1, s.val2)
category(s::Label) = s.cat

# TODO: add tests for this
function val_to_str(s::Label)
  if nvals(category(s)) == 1
    return val_to_str(category(s), val(s))
  end
  return val_to_str(category(s), vals(s))
end

fusion_rule(C::DynamicCategory, vals...) = fusion_rule(static(C),vals...)

function ⊗(s1::Label, s2::Label)
  name(s1) == name(s2) ||
    error("Cannot fuse sector labels with different names \"$(name(s1))\", \"$(name(s2))\"")
  category(s1) == category(s2) ||
    error("Labels with matching name \"$(name(s1))\" cannot have different category types")
  return [
    Label(name(s1), v, category(s1)) for v in fusion_rule(category(s1), vals(s1), vals(s2))
  ]
end
Base.:(*)(s1::Label, s2::Label) = ⊗(s1, s2)

function Base.show(io::IO, l::Label)
  if nvals(category(l)) == 1
    return print(io, "Label(\"", name(l), "\",", val(l), ",", category(l), ")")
  end
  return print(io, "Label(\"", name(l), "\",", vals(l), ",", category(l), ")")
end
