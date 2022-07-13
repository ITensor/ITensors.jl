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
function _op(::OpName"Id", ::SiteType"Qudit"; dim::Tuple=(2,))
  d = dim[1]
  mat = zeros(d, d)
  for k in 1:d
    mat[k, k] = 1.0
  end
  return mat
end

function _op(::OpName"I", st::SiteType"Qudit"; kwargs...)
  return _op(OpName"Id"(), st; kwargs...)
end

function _op(::OpName"Adag", ::SiteType"Qudit"; dim::Tuple=(2,))
  d = dim[1]
  mat = zeros(d, d)
  for k in 1:(d - 1)
    mat[k + 1, k] = √k
  end
  return mat
end
_op(::OpName"adag", st::SiteType"Qudit"; kwargs...) = _op(OpName"Adag"(), st; kwargs...)
_op(::OpName"a†", st::SiteType"Qudit"; kwargs...) = _op(OpName"Adag"(), st; kwargs...)

function _op(::OpName"A", ::SiteType"Qudit"; dim::Tuple=(2,))
  d = dim[1]
  mat = zeros(d, d)
  for k in 1:(d - 1)
    mat[k, k + 1] = √k
  end
  return mat
end
_op(::OpName"a", st::SiteType"Qudit"; kwargs...) = _op(OpName"A"(), st; kwargs...)

function _op(::OpName"N", ::SiteType"Qudit"; dim::Tuple=(2,))
  d = dim[1]
  mat = zeros(d, d)
  for k in 1:d
    mat[k, k] = k - 1
  end
  return mat
end
_op(::OpName"n", st::SiteType"Qudit"; kwargs...) = _op(OpName"N"(), st; kwargs...)

# two-body operators 
function _op(::OpName"ab", st::SiteType"Qudit"; dim::Tuple=(2, 2))
  return kron(_op(OpName("a"), st; dim=(dim[1],)), _op(OpName("a"), st; dim=(dim[2],)))
end

function _op(::OpName"a†b", st::SiteType"Qudit"; dim::Tuple=(2, 2))
  return kron(_op(OpName("a†"), st; dim=(dim[1],)), _op(OpName("a"), st; dim=(dim[2],)))
end

function _op(::OpName"ab†", st::SiteType"Qudit"; dim::Tuple=(2, 2))
  return kron(_op(OpName("a"), st; dim=(dim[1],)), _op(OpName("a†"), st; dim=(dim[2],)))
end

function _op(::OpName"a†b†", st::SiteType"Qudit"; dim::Tuple=(2, 2))
  return kron(_op(OpName("a†"), st; dim=(dim[1],)), _op(OpName("a†"), st; dim=(dim[2],)))
end

# interface
function op(on::OpName, st::SiteType"Qudit", s::Index...)
  rs = reverse([s...])
  d⃗ = dim.(Tuple(rs))
  opmat = _op(on, st; dim=d⃗)
  return ITensors.itensor(opmat, prime.(rs)..., dag.(rs)...)
end

# Zygote
@non_differentiable _op(::OpName"ab", ::SiteType"Qudit")
@non_differentiable _op(::OpName"a†b", ::SiteType"Qudit")
@non_differentiable _op(::OpName"ab†", ::SiteType"Qudit")
@non_differentiable _op(::OpName"a†b†", ::SiteType"Qudit")
@non_differentiable _op(::OpName"a", ::SiteType"Qudit")
@non_differentiable _op(::OpName"a†", ::SiteType"Qudit")
@non_differentiable _op(::OpName"N", ::SiteType"Qudit")
