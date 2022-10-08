"""
    space(::SiteType"Qudit";
          dim = 2,
          conserve_qns = false,
          conserve_number = false,
          qnname_number = "Number")

Create the Hilbert space for a site of type "Qudit".

Optionally specify the conserved symmetries and their quantum number labels.
"""
function ITensors.space(
  ::SiteType"Qudit";
  dim=2,
  conserve_qns=false,
  conserve_number=conserve_qns,
  qnname_number="Number",
)
  if conserve_number
    return [QN(qnname_number, n - 1) => 1 for n in 1:dim]
  end
  return dim
end

function ITensors.val(::ValName{N}, ::SiteType"Qudit") where {N}
  return parse(Int, String(N)) + 1
end

function ITensors.state(::StateName{N}, ::SiteType"Qudit", s::Index) where {N}
  n = parse(Int, String(N))
  st = zeros(dim(s))
  st[n + 1] = 1.0
  return itensor(st, s)
end

# one-body operators
function op(::OpName"Id", ::SiteType"Qudit", ds::Int...)
  d = prod(ds)
  return Matrix(1.0I, d, d)
end
op(on::OpName"I", st::SiteType"Qudit", ds::Int...) = op(alias(on), st, ds...)
op(on::OpName"F", st::SiteType"Qudit", ds::Int...) = op(OpName"Id"(), st, ds...)

function op(::OpName"Adag", ::SiteType"Qudit", d::Int)
  mat = zeros(d, d)
  for k in 1:(d - 1)
    mat[k + 1, k] = √k
  end
  return mat
end
op(on::OpName"adag", st::SiteType"Qudit", d::Int) = op(alias(on), st, d)
op(on::OpName"a†", st::SiteType"Qudit", d::Int) = op(alias(on), st, d)

function op(::OpName"A", ::SiteType"Qudit", d::Int)
  mat = zeros(d, d)
  for k in 1:(d - 1)
    mat[k, k + 1] = √k
  end
  return mat
end
op(on::OpName"a", st::SiteType"Qudit", d::Int) = op(alias(on), st, d)

function op(::OpName"N", ::SiteType"Qudit", d::Int)
  mat = zeros(d, d)
  for k in 1:d
    mat[k, k] = k - 1
  end
  return mat
end
op(on::OpName"n", st::SiteType"Qudit", d::Int) = op(alias(on), st, d)

# two-body operators 
function op(::OpName"ab", st::SiteType"Qudit", d1::Int, d2::Int)
  return kron(op(OpName("a"), st, d1), op(OpName("a"), st, d2))
end

function op(::OpName"a†b", st::SiteType"Qudit", d1::Int, d2::Int)
  return kron(op(OpName("a†"), st, d1), op(OpName("a"), st, d2))
end

function op(::OpName"ab†", st::SiteType"Qudit", d1::Int, d2::Int)
  return kron(op(OpName("a"), st, d1), op(OpName("a†"), st, d2))
end

function op(::OpName"a†b†", st::SiteType"Qudit", d1::Int, d2::Int)
  return kron(op(OpName("a†"), st, d1), op(OpName("a†"), st, d2))
end

# interface
function op(on::OpName, st::SiteType"Qudit", s1::Index, s_tail::Index...; kwargs...)
  rs = reverse((s1, s_tail...))
  ds = dim.(rs)
  opmat = op(on, st, ds...; kwargs...)
  return itensor(opmat, prime.(rs)..., dag.(rs)...)
end

function op(on::OpName, st::SiteType"Qudit"; kwargs...)
  return error("`op` can't be called without indices or dimensions.")
end

# Zygote
@non_differentiable op(::OpName"ab", ::SiteType"Qudit", ::Int, ::Int)
@non_differentiable op(::OpName"a†b", ::SiteType"Qudit", ::Int, ::Int)
@non_differentiable op(::OpName"ab†", ::SiteType"Qudit", ::Int, ::Int)
@non_differentiable op(::OpName"a†b†", ::SiteType"Qudit", ::Int, ::Int)
@non_differentiable op(::OpName"a", ::SiteType"Qudit", ::Int)
@non_differentiable op(::OpName"a†", ::SiteType"Qudit", ::Int)
@non_differentiable op(::OpName"N", ::SiteType"Qudit", ::Int)
