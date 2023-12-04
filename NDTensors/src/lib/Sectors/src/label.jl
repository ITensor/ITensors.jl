#
# Label - single symmetry sector label
# 

struct Label
  name::String7
  val1::Half{Int}
  val2::Half{Int}
  cat::Category
end

Label() = Label("", 0, 0, Category())
Label(name::AbstractString, val::Number, cat=U(1))  = Label(name,val,0,cat)
Label(name::AbstractString, val::AbstractString, cat=U(1))  = Label(name,string_to_val(cat, val),0,cat)

Label(name::AbstractString, vals::Tuple, cat=U(1)) = Label(name, vals[1], vals[2], cat)

name(s::Label) = s.name
function val(s::Label) 
  s.val2 != 0 && error("Second value of Label is non-zero = ($(s.val1),$(s.val2))")
  return s.val1
end
vals(s::Label) = (s.val1,s.val2)
category(s::Label) = s.cat

string_to_val(C::CategoryName,v::AbstractString) = error("String values not implemented for category $(name(C))")
string_to_val(C::Category,v) = string_to_val(CategoryName(C),v)

val_to_str(::Any, val) = string(val)
function val_to_str(s::Label) 
  if nvals(category(s)) == 1
    return val_to_str(Val(category(s).basename), val(s))
  else
    return val_to_str(Val(category(s).basename), vals(s))
  end
end

isactive(s::Label) = isactive(category(s))

zero(s::Label) = Label(name(s), 0, category(s))

length(s::Label) = 1

const InactiveVal = Label()

function ⊗(s1::Label, s2::Label)
  name(s1) == name(s2) ||
    error("Cannot fuse sector labels with different names \"$(name(s1))\", \"$(name(s2))\"")
  category(s1) == category(s2) ||
    error("Labels with matching name \"$(name(s1))\" cannot have different category types")
  return [Label(name(s1), v, category(s1)) for v in ⊗(category(s1), vals(s1), vals(s2))]
end
*(s1::Label, s2::Label) = ⊗(s1, s2)

function show(io::IO, l::Label)
  if nvals(category(l)) == 1
    return print(io, "Label(\"", name(l), "\",", val(l), ",", category(l), ")")
  else
    return print(io, "Label(\"", name(l), "\",", vals(l), ",", category(l), ")")
  end
end

