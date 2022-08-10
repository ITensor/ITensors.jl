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
function op(::OpName"Id", ::SiteType"Qudit", dims::Tuple)
  d = prod(dims)
  return Matrix(1.0I, d, d)
end
op(on::OpName"I", st::SiteType"Qudit", dims::Tuple) = op(alias(on), st, dims)

function op(::OpName"Adag", ::SiteType"Qudit", dims::Tuple)
  d = dims[1]
  mat = zeros(d, d)
  for k in 1:(d - 1)
    mat[k + 1, k] = √k
  end
  return mat
end
op(on::OpName"adag", st::SiteType"Qudit", dims::Tuple) = op(alias(on), st, dims)
op(on::OpName"a†", st::SiteType"Qudit", dims::Tuple) = op(alias(on), st, dims)

function op(::OpName"A", ::SiteType"Qudit", dims::Tuple)
  d = dims[1]
  mat = zeros(d, d)
  for k in 1:(d - 1)
    mat[k, k + 1] = √k
  end
  return mat
end
op(on::OpName"a", st::SiteType"Qudit", dims::Tuple) = op(alias(on), st, dims)

function op(::OpName"N", ::SiteType"Qudit", dims::Tuple)
  d = dims[1]
  mat = zeros(d, d)
  for k in 1:d
    mat[k, k] = k - 1
  end
  return mat
end
op(on::OpName"n", st::SiteType"Qudit", dims::Tuple) = op(alias(on), st, dims)

# two-body operators 
function op(::OpName"ab", st::SiteType"Qudit", dims::Tuple)
  return kron(op(OpName("a"), st, (dims[1],)), op(OpName("a"), st, (dims[2],)))
end

function op(::OpName"a†b", st::SiteType"Qudit", dims::Tuple)
  return kron(op(OpName("a†"), st, (dims[1],)), op(OpName("a"), st, (dims[2],)))
end

function op(::OpName"ab†", st::SiteType"Qudit", dims::Tuple)
  return kron(op(OpName("a"), st, (dims[1],)), op(OpName("a†"), st, (dims[2],)))
end

function op(::OpName"a†b†", st::SiteType"Qudit", dims::Tuple)
  return kron(op(OpName("a†"), st, (dims[1],)), op(OpName("a†"), st, (dims[2],)))
end

# interface
function op(on::OpName, st::SiteType"Qudit", s::Index...)
  rs = reverse([s...])
  d⃗ = dim.(Tuple(rs))
  opmat = op(on, st, d⃗)
  return ITensors.itensor(opmat, prime.(rs)..., dag.(rs)...)
end

# Zygote
@non_differentiable op(::OpName"ab", ::SiteType"Qudit", ::Tuple)
@non_differentiable op(::OpName"a†b", ::SiteType"Qudit", ::Tuple)
@non_differentiable op(::OpName"ab†", ::SiteType"Qudit", ::Tuple)
@non_differentiable op(::OpName"a†b†", ::SiteType"Qudit", ::Tuple)
@non_differentiable op(::OpName"a", ::SiteType"Qudit", ::Tuple)
@non_differentiable op(::OpName"a†", ::SiteType"Qudit", ::Tuple)
@non_differentiable op(::OpName"N", ::SiteType"Qudit", ::Tuple)
