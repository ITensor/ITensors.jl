using InlineStrings: String7
using HalfIntegers: Half

"""
Representation of a category or group, storing its
name and the dimension of its defining representation.
One can also specify a quantum deformation "level".

For example, Category("SU",2) is the group SU(2). 
Category("Fib") is the Fibonacci category. 
Category("SU",2,3) is the "quantum group" SU(2) level 3.
"""
struct Category
  basename::String7
  N::Int
  level::Int
end

Category(basename, N::Int=0) = Category(basename, N, 0)

Category() = Category("")

basename(C::Category) = C.basename
groupdim(C::Category) = C.N
level(C::Category) = C.level

isactive(C::Category) = length(C.basename) > 0

function name(C::Category)
  if groupdim(C) == 0
    return C.basename
  elseif level(C) == 0
    return "$(C.basename)($(groupdim(C)))"
  else
    return "$(C.basename)($(groupdim(C)))_$(level(C))"
  end
  return error("Unexpected case")
end

Base.show(io::IO, C::Category) = print(io, name(C))

nvals(::Any) = 1
nvals(C::Category) = nvals(CategoryName(C))

function fusion_rule(C::Category, a, b)
  if nvals(C) == 1
    return fusion_rule(CategoryName(C), a[1], b[1])
  end
  return fusion_rule(CategoryName(C), a, b)
end

#
# Version of Category type but with
# basename held statically as a 
# type parameter. Mostly for internal use.
#
struct CategoryName{Name}
  N::Int
  level::Int
end

CategoryName(C::Category) = CategoryName{basename(C)}(groupdim(C), level(C))

basename(C::CategoryName{Name}) where {Name} = Name
groupdim(C::CategoryName) = C.N
level(C::CategoryName) = C.level
name(C::CategoryName) = name(Category(C))

"""
Convenience macro for value dispatch on names 
of Category objects. Writing (::CategoryName"Name", ...)
in function arguments allows Val-based dispatch based on
the name value.
"""
macro CategoryName_str(s)
  return :(CategoryName{$(Expr(:quote, String7(s)))})
end

Category(C::CategoryName) = Category(basename(C), groupdim(C), level(C))
