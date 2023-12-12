using InlineStrings: String7
using HalfIntegers: Half

default_category_name_type() = String7

"""
Representation of a category or group, storing its
name and the dimension of its defining representation.
One can also specify a quantum deformation "level".

For example, Category("SU",2) is the group SU(2). 
Category("Fib") is the Fibonacci category. 
Category("SU",2,3) is the "quantum group" SU(2) level 3.
"""
struct Category{Name}
  basename::Name
  groupdim::Int
  level::Int

  global function _Category(basename, groupdim, level)
    return new{typeof(basename)}(basename, groupdim, level)
  end
end

function Category(
  basename="", N::Int=0, level::Int=0; name_type=default_category_name_type()
)
  return _Category(convert(name_type, basename), N, level)
end

basename(C::Category) = C.basename
groupdim(C::Category) = C.groupdim
level(C::Category) = C.level
name_type(C::Category{Name}) where {Name} = Name

function name(C::Category)
  if groupdim(C) == 0
    return basename(C)
  elseif level(C) == 0
    return "$(basename(C))($(groupdim(C)))"
  else
    return "$(basename(C))($(groupdim(C)))_$(level(C))"
  end
  return error("Unexpected case")
end

Base.show(io::IO, C::Category) = print(io, name(C))

struct CategoryName{Name} end

CategoryName(S) = CategoryName{S}()

basename(C::CategoryName{Name}) where {Name} = Name

# TODO:
# This method makes `basename` return the same type (and value) 
# of the dynamic and static versions of a Category
# Is that the behavior we want? (Otherwise basename(D) != basename(static(D)).)
# May be enough to just make CategoryName's print whatever Name they hold.
basename(C::Category{CName}) where {CName<:CategoryName} = basename(CName())

function static(C::Category)
  CName = CategoryName{basename(C)}
  return Category(CName(), groupdim(C), level(C); name_type=CName)
end

dynamic(C::Category) = Category(basename(C), groupdim(C), level(C))

is_static(C::Category) = (name_type(C) <: CategoryName)

is_dynamic(C::Category) = !is_static(C)

"""
Convenience macro for value dispatch on names 
of Category objects. Writing (::CategoryType"Name", ...)
in function arguments allows Val-based dispatch based on
the name value.
"""
# TODO: how to avoid using default_category_name_type() here?
# Maybe always convert to Symbol?
macro CategoryType_str(s)
  return :(Category{$(Expr(:quote, CategoryName{default_category_name_type()(s)}))})
end

#
# Methods that can be customized for specific categories 
#
const DynamicCategory = Category{<:AbstractString}

# TODO:
# Tried this... could something like this work?
#const StaticCategory = Category{<:CategoryName}
#Base.convert(::Type{StaticCategory}, D::DynamicCategory) = static(D)

nvals(::Category) = 1
nvals(C::DynamicCategory) = nvals(static(C))

function fusion_rule(C::Category, a, b)
  return error("fusion_rule not implemented for category $(basename(C))")
end
function fusion_rule(C::DynamicCategory, a, b)
  # TODO: improve... maybe don't use 
  # nvals and just pass a and b?
  if nvals(C) == 1
    return fusion_rule(static(C), a[1], b[1])
  end
  return fusion_rule(static(C), a, b)
end

function str_to_val(C::Category, s)
  return error("String values not implemented for category $(basename(C))")
end
str_to_val(C::DynamicCategory, s) = str_to_val(static(C), s)

val_to_str(::Category, val) = string(val)
val_to_str(C::DynamicCategory, val) = val_to_str(static(C), val)

istrivial(::Category, values) = all(iszero, values)
