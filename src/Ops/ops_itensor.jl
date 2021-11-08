function itensor(I::UniformScaling, is...)
  return ITensor(I, is...)
end

function ITensor(I::UniformScaling, is...)
  return ITensor(I(isqrt(dim(is))), is...)
end

# Using ITensors.jl definitions
function _ITensor(
  which_op::AbstractString, sites::Tuple, params::NamedTuple, s::Vector{<:Index}
)
  return op(which_op, s, sites; params...)
end

function _ITensor(
  which_op::Union{AbstractMatrix,UniformScaling},
  sites::Tuple,
  params::NamedTuple,
  s::Vector{<:Index},
)
  sₙ = s[collect(sites)]
  return itensor(which_op, sₙ', dag(sₙ))
end

# TODO: Get this more directly from the `ITensors.op` definitions.
Matrix(o::Op, s::Vector{<:Index}) = array(ITensor(o, s))

function ITensor(o::Op, s::Vector{<:Index})
  return _ITensor(Tuple(o)..., s)
end

function hassamesites(o)
  if length(o) ∈ (0, 1)
    return true
  end
  return reduce(issetequal, Ops.sites.(o))
end

function Matrix(o::∑, s::Vector{<:Index})
  if hassamesites(o)
    return o.f([Matrix(arg, s) for arg in o])
  end
  return error("Not implemented")
end

function Matrix(o::∏, s::Vector{<:Index})
  if hassamesites(o)
    return o.f([Matrix(arg, s) for arg in o])
  end
  return error("Not implemented")
end

function Matrix(o::α, s::Vector{<:Index})
  return coefficient(o) * Matrix(Ops.op(o), s)
end

function Matrix(o::Applied{typeof(exp)}, s::Vector{<:Index})
  return o.f(Matrix(Ops.op(o), s))
end

function Matrix(o::Applied{typeof(adjoint)}, s::Vector{<:Index})
  return o.f(Matrix(Ops.op(o), s))
end

function ITensor(o::Applied, s::Vector{<:Index})
  m = Matrix(o, s)
  return ITensor(Op(m, Ops.sites(o)), s)
end
