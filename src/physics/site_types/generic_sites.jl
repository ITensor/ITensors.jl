function op(::OpName"Id", ::SiteType"Generic", s1::Index, sn::Index...; eltype=Float64)
  s = (s1, sn...)
  n = prod(dim.(s))
  return itensor(Matrix(one(eltype) * I, n, n), prime.(s)..., dag.(s)...)
end

function op(::OpName"I", st::SiteType"Generic", s::Index...; kwargs...)
  return op(OpName("Id"), st, s...; kwargs...)
end

function op(::OpName"F", st::SiteType"Generic", s::Index; kwargs...)
  return op(OpName("Id"), st, s; kwargs...)
end

function default_random_matrix(eltype::Type, s::Index...)
  n = prod(dim.(s))
  return randn(eltype, n, n)
end

# Haar-random unitary
#
# Reference:
# Section 4.6
# http://math.mit.edu/~edelman/publications/random_matrix_theory.pdf
function op(
  ::OpName"RandomUnitary",
  ::SiteType"Generic",
  s1::Index,
  sn::Index...;
  eltype=ComplexF64,
  random_matrix=default_random_matrix(eltype, s1, sn...),
)
  s = (s1, sn...)
  Q, _ = NDTensors.qr_positive(random_matrix)
  return itensor(Q, prime.(s)..., dag.(s)...)
end

function op(::OpName"randU", st::SiteType"Generic", s::Index...; kwargs...)
  return op(OpName("RandomUnitary"), st, s...; kwargs...)
end
