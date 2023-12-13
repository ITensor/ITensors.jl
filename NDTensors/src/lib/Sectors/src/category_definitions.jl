
#
# 2D rotation group U(1)
#

U(N::Int) = Category("U", N)

fusion_rule(::CategoryType"U", n1, n2) = [(n1 + n2)]

#
# Cyclic group Ƶₙ
#

Z(N::Int) = Category("Z", N)

fusion_rule(C::CategoryType"Z", n1, n2) = [(n1 + n2) % groupdim(C)]

#
# SUd(N)
# Special unitary group SU(N)
# using dimensions as labels
#

SUd(N::Int, k=0) = Category("SUd", N, k)

function fusion_rule(C::CategoryType"SUd", d1, d2)
  if groupdim(C) != 2
    error(
      "Only SUd(2) and SUd(2)_k currently supported [input was SUd($(groupdim(C)),$(level(C)))]",
    )
  end
  if level(C) == 0
    return collect((abs(d1 - d2) + 1):2:(d1 + d2 - 1))
  else
    error("level > 0 not yet supported for category SUd")
  end
end

#
# Special unitary group SU(N)
# and quantum groups SU(N)ₖ
#

SU(N::Int, k=0) = Category("SU", N, k)

function fusion_rule(C::CategoryType"SU", j1, j2)
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
# while still organizing spaces into
# well-defined j values
#

SUz(N::Int) = Category("SUz", N)

function fusion_rule(C::CategoryType"SUz", jm1::Tuple, jm2::Tuple)
  @assert groupdim(C) == 2
  @assert level(C) == 0
  j1, m1 = jm1
  j2, m2 = jm2
  return [(J, m1 + m2) for J in abs(m1 + m2):(j1 + j2)]
end

nvals(C::CategoryType"SUz") = 2

#
# Fibonacci category
#

const Fib = Category("Fib")

# Fusion rules of subcategory containing
# {0,1} of A_4 i.e. su(2)₃
# (see arxiv:2008.08598)
fusion_rule(::CategoryType"Fib", a1, a2) = fusion_rule(SU(2, 3), a1, a2)

val_to_str(::CategoryType"Fib", a) = ("1", "τ")[first(a) + 1]
str_to_val(::CategoryType"Fib", a::AbstractString) = (a == "τ") ? 1 : 0

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
fusion_rule(::CategoryType"Ising", a1, a2) = fusion_rule(SU(2, 2), a1, a2)

val_to_str(::CategoryType"Ising", a) = ("1", "σ", "ψ")[Int(2 * a + 1)]

function str_to_val(::CategoryType"Ising", s::AbstractString)
  for (a, v) in enumerate(("1", "σ", "ψ"))
    (v == s) && return (a - 1)//2
  end
  return error("Unrecognized string \"$s\" for Ising category")
end
