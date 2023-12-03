
struct Category
  basename::String7
  N::Int
  level::Int
end

Category(basename, N::Int=0) = Category(basename, N, 0)

Category() = Category("")

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

"""
Convenience macro for value dispatch on names 
of Category objects. Writing (::CategoryName"Name", ...)
in function arguments allows Val-based dispatch based on
the name value.
"""
macro CategoryName_str(s)
  return :(Val{$(Expr(:quote, String7(s)))})
end

nvals(::Any) = 1
nvals(C::Category) = nvals(Val(C.basename))

function ⊗(C::Category, a, b) 
  if nvals(C)==1
    return ⊗(Val(C.basename), C, a[1], b[1])
  else
    return ⊗(Val(C.basename), C, a, b)
  end
end


#
# 2D rotation group U(1)
#

U(N::Int) = Category("U", N)

⊗(::CategoryName"U", C, n1, n2) = [(n1 + n2)]

#
# Cyclic group Ƶₙ
#

Z(N::Int) = Category("Z", N)

⊗(::CategoryName"Z", C, n1, n2) = [(n1 + n2) % groupdim(C)]

#
# SUd(N)
# Special unitary group SU(N)
# using dimensions as labels
#


SUd(N::Int, k=0) = Category("SUd", N, k)

function ⊗(::CategoryName"SUd", C, d1, d2)
  if groupdim(C) != 2
    error(
      "Only SUd(2) and SUd(2)_k currently supported [input was SUd($(groupdim(C)),$(level(C)))]",
    )
  end
  if level(C) == 0
    #@show collect((abs(d1 - d2)+1):(d1+d2-1))
    return collect((abs(d1 - d2)+1):2:(d1+d2-1))
  else
    error("level > 0 not yet supported for category SUd")
  end
end

#
# Special unitary group SU(N)
# and quantum groups SU(N)ₖ
#

SU(N::Int, k=0) = Category("SU", N, k)

function ⊗(::CategoryName"SU", C, j1, j2)
  if groupdim(C) != 2
    error(
      "Only SU(2) and SU(2)_k currently supported [input was SU($(groupdim(C)),$(level(C)))]",
    )
  end
  if level(C) == 0
    return collect(abs(j1 - j2):(j1 + j2))
  else
    return collect(abs(j1 - j2):min(level(C) - j1 - j2, j1 + j2))
  end
end

#
# Special unitary group SU(N)
# but only conserving U(1) (Sz) subgroup
#

SUz(N::Int) = Category("SUz", N)

function ⊗(::CategoryName"SUz", C, jm1::Tuple, jm2::Tuple)
  @assert groupdim(C) == 2
  @assert level(C) == 0
  j1,m1 = jm1
  j2,m2 = jm2
  return [(J,m1+m2) for J in abs(m1+m2):(j1+j2)]
end

nvals(::CategoryName"SUz") = 2

#
# Fibonacci category
#

const Fib = Category("Fib")

# Fusion rules of subcategory containing
# {0,1} of A_4 i.e. su(2)₃
# (see arxiv:2008.08598)
⊗(::CategoryName"Fib", C, a1, a2) = ⊗(SU(2, 3), a1, a2)

val_to_str(::CategoryName"Fib", a) = ("1", "τ")[a + 1]
string_to_val(::CategoryName"Fib", a::AbstractString) = (a == "τ") ? 1 : 0

#
# Ising category
#

const Ising = Category("Ising")

# Thinking of ⊗ as "+" on the a values:
# 0+a = a  (same as 1⊗a = a
# 1+1 = 0  (same as  ψ⊗ψ = 1)
# 1/2+1/2 = 0 ⊕ 1  (same as  σ⊗σ = 1 + ψ)
# 1+1/2 = 1/2  (same as  ψ⊗σ = σ)
# i.e. fusion rules are su(2)₂ a.k.a. A_3
# but with different Frobenius-Schur sign
# (see arxiv:2008.08598)
⊗(::CategoryName"Ising", C, a1, a2) = ⊗(SU(2, 2), a1, a2)

val_to_str(::CategoryName"Ising", a) = ("1", "σ", "ψ")[Int(2 * a + 1)]

function string_to_val(::CategoryName"Ising", s::AbstractString)
  for (a, v) in enumerate(("1", "σ", "ψ"))
    (v == s) && return (a - 1)//2
  end
  return error("Unrecognized string \"$s\" for Ising category")
end

