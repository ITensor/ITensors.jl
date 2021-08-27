
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
  return st
end

function _op(::OpName"Id", ::SiteType"Qudit"; dim=2)
  mat = zeros(dim, dim)
  for k in 1:dim
    mat[k, k] = 1.0
  end
  return mat
end

function _op(::OpName"Adag", ::SiteType"Qudit"; dim=2)
  mat = zeros(dim, dim)
  for k in 1:(dim - 1)
    mat[k + 1, k] = √k
  end
  return mat
end
_op(::OpName"adag", st::SiteType"Qudit"; kwargs...) = _op(OpName"Adag"(), st; kwargs...)
_op(::OpName"a†", st::SiteType"Qudit"; kwargs...) = _op(OpName"Adag"(), st; kwargs...)

function _op(::OpName"A", ::SiteType"Qudit"; dim=2)
  mat = zeros(dim, dim)
  for k in 1:(dim - 1)
    mat[k, k + 1] = √k
  end
  return mat
end
_op(::OpName"a", st::SiteType"Qudit"; kwargs...) = _op(OpName"A"(), st; kwargs...)

function _op(::OpName"N", ::SiteType"Qudit"; dim=2)
  mat = zeros(dim, dim)
  for k in 1:dim
    mat[k, k] = k - 1
  end
  return mat
end
_op(::OpName"n", st::SiteType"Qudit"; kwargs...) = _op(OpName"N"(), st; kwargs...)

function ITensors.op(on::OpName, st::SiteType"Qudit", s::Index)
  return itensor(_op(on, st; dim=dim(s)), s', dag(s))
end
