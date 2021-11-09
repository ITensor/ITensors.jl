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

function hassamesites(o)
  if length(o) ∈ (0, 1)
    return true
  end
  return reduce(issetequal, Ops.sites.(o))
end

function ITensor(o::Op, s::Vector{<:Index})
  return _ITensor(Tuple(o)..., s)
end

# Extend the operator `o` to the sites `n`
# by filling the rest of the sites with `op`.
function insert_ids(o, n)
  insert_n = setdiff(n, Ops.sites(o))
  for i in insert_n
    o *= Op(I, i)
  end
  return o
end

# TODO: Merge these two.
function insert_ids(o::∏)
  n = Ops.sites(o)
  return ∏([insert_ids(oₙ, n) for oₙ in o])
end
function insert_ids(o::∑)
  n = Ops.sites(o)
  return ∑([insert_ids(oₙ, n) for oₙ in o])
end

# TODO: Does this work for fermions?
function ITensor(o::∏, s::Vector{<:Index})
  o_id = insert_ids(o)
  ∏layers = ∏([∏{ITensor}(oₙ, s) for oₙ in o_id])
  res = ITensor(Op(I, Ops.sites(o)), s)
  for layer in reverse(∏layers)
    res = layer(res)
  end
  return res
end

function ITensor(o::∑{ITensor})
  res = ITensor()
  for oₙ in o
    res += oₙ
  end
  return res
end

function ITensor(o::∑, s::Vector{<:Index})
  o_id = insert_ids(o)
  ∑layers = ∑([ITensor(oₙ, s) for oₙ in o_id])
  return ITensor(∑layers)
  ## if hassamesites(o)
  ##   return o.f([ITensor(arg, s) for arg in o])
  ## end
  ## return error("Trying to make an ITensor from operator expression $o. When making an ITensor from a sum of operators, the operators need to have the same sites.")
end

function ITensor(o::α, s::Vector{<:Index})
  return coefficient(o) * ITensor(Ops.op(o), s)
end

function ITensor(o::Applied{typeof(exp)}, s::Vector{<:Index})
  return o.f(ITensor(Ops.op(o), s))
end

itensor_adjoint(T::ITensor) = swapprime(dag(T), 0 => 1)

function ITensor(o::Applied{typeof(adjoint)}, s::Vector{<:Index})
  return itensor_adjoint(ITensor(Ops.op(o), s))
end

function ITensor(o::Applied, s::Vector{<:Index})
  return error("Trying to make ITensor from expression $(o), not yet implemented.")
end

∏{ITensor}(o::∏, s::Vector{<:Index}) = ∏([ITensor(oₙ, s) for oₙ in o])
∏{ITensor}(o::Union{Op,Applied}, s::Vector{<:Index}) = ∏{ITensor}(∏([o]), s)
∑{ITensor}(o::∑, s::Vector{<:Index}) = ∑([ITensor(oₙ, s) for oₙ in o])
∑{ITensor}(o::Union{Op,Applied}, s::Vector{<:Index}) = ∑{ITensor}(∑([o]), s)

(o::∏{ITensor})(x; kwargs...) = apply(o, x; kwargs...)
# Apply it in reverse to follow the linear algebra convention:
# (O₁ * O₂)|x⟩ = O₁ * (O₂|x⟩)
apply(o::∏{ITensor}, x; kwargs...) = apply([oₙ for oₙ in reverse(o)], x; kwargs...)
