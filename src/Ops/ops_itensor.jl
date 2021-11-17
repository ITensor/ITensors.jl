
#
# General ITensor functions
#

function itensor(I::UniformScaling, is...)
  return ITensor(I, is...)
end

function ITensor(I::UniformScaling, is...)
  return ITensor(I(isqrt(dim(is))), is...)
end

#
# Utils
#

# Extend the operator `o` to the sites `n`
# by filling the rest of the sites with `op`.
function insert_ids(o, n)
  insert_n = setdiff(n, Ops.sites(o))
  ∏o = ∏([o])
  for i in insert_n
    ∏o *= Op(I, i)
  end
  return ∏o
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

#
# Lazy ITensor to ITensor
#

function ITensor(o::∑{ITensor})
  res = ITensor()
  for oₙ in o
    res += oₙ
  end
  return res
end

#
# Op to Matrix constructors
#

function Base.Matrix(o::Op, s::Vector{<:Index})
  which_op = Ops.which_op(o)
  op_sites = Ops.sites(o)
  op_params = Ops.params(o)
  return Matrix(which_op, op_sites, s; op_params...)
end

function Base.Matrix(which_op::AbstractMatrix, op_sites, s::Vector{<:Index}; op_params...)
  d = dim([s[n] for n in op_sites])
  @assert size(which_op, 1) == size(which_op, 2) == d
  return which_op
end

function Base.Matrix(which_op::UniformScaling, op_sites, s::Vector{<:Index}; op_params...)
  d = dim([s[n] for n in op_sites])
  return Matrix(which_op, d, d)
end

function Base.Matrix(which_op, op_sites, s::Vector{<:Index}; op_params...)
  s⃗ = [s[n] for n in op_sites]
  # TODO: Handle case with multiple common tags
  sitetype = SiteType(only(ITensors.commontags(s⃗)))
  return op(OpName(which_op), sitetype; op_params...)
end

function Matrix(o::α, s::Vector{<:Index})
  return coefficient(o) * Matrix(Ops.op(o), s)
end

function Matrix(o::Applied{typeof(exp)}, s::Vector{<:Index})
  return o.f(Matrix(Ops.op(o), s))
end

#
# Prod Op to Matrix
#

## function _Matrix_unique_sites(o::∏{Op}, s::Vector{<:Index})
##   @show o
##   Ms = [Matrix(oₙ, s) for oₙ in o]
##   @show Ms
##   @show reverse(Ms)
##   return reduce(kron, reverse(Ms))
## end
## 
## function Matrix(o::∏, s::Vector{<:Index})
##   @show o
##   o_id = insert_ids(o)
##   @show o_id
##   ∏layers = ∏([_Matrix_unique_sites(oₙ, s) for oₙ in o_id])
##   @show ∏layers
##   return LazyApply.materialize(∏layers)
## end

#
# Op to ITensor constructors
#

function ITensors.ITensor(o::Op, s::Vector{<:Index})
  n⃗ = Ops.sites(o)
  s⃗ = [s[n] for n in n⃗]
  return itensor(Matrix(o, s), s⃗', dag(s⃗))
end

function ITensor(o::Applied, s::Vector{<:Index})
  fm = Matrix(o, s)
  return ITensor(Ops.set_which_op(o, fm), s)
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

#
# Op to Lazy ITensor
#

∏{ITensor}(o::∏, s::Vector{<:Index}) = ∏([ITensor(oₙ, s) for oₙ in o])
∏{ITensor}(o::Union{Op,Applied}, s::Vector{<:Index}) = ∏{ITensor}(∏([o]), s)
∑{ITensor}(o::∑, s::Vector{<:Index}) = ∑([ITensor(oₙ, s) for oₙ in o])
∑{ITensor}(o::Union{Op,Applied}, s::Vector{<:Index}) = ∑{ITensor}(∑([o]), s)

#
# Apply a product of ITensors
#

(o::∏{ITensor})(x; kwargs...) = apply(o, x; kwargs...)
# Apply it in reverse to follow the linear algebra convention:
# (O₁ * O₂)|x⟩ = O₁ * (O₂|x⟩)
apply(o::∏{ITensor}, x; kwargs...) = apply([oₙ for oₙ in reverse(o)], x; kwargs...)

#
# Deprecated
#

## function ITensor(o::∑, s::Vector{<:Index})
##   o_id = insert_ids(o)
##   ∑layers = ∑([ITensor(oₙ, s) for oₙ in o_id])
##   return ITensor(∑layers)
## end
## 
## itensor_adjoint(T::ITensor) = swapprime(dag(T), 0 => 1)
## 
## function ITensor(o::Applied{typeof(adjoint)}, s::Vector{<:Index})
##   return itensor_adjoint(ITensor(Ops.op(o), s))
## end
## 
## function ITensor(o::Applied, s::Vector{<:Index})
##   return error("Trying to make ITensor from expression $(o), not yet implemented.")
## end
## 
## function hassamesites(o)
##   if length(o) ∈ (0, 1)
##     return true
##   end
##   return reduce(issetequal, Ops.sites.(o))
## end
## 
## function _ITensor(
##   which_op::AbstractString, sites::Tuple, params::NamedTuple, s::Vector{<:Index}
## )
##   return op(which_op, s, sites; params...)
## end
## 
## function _ITensor(
##   which_op::Union{AbstractMatrix,UniformScaling},
##   sites::Tuple,
##   params::NamedTuple,
##   s::Vector{<:Index},
## )
##   sₙ = s[collect(sites)]
##   return itensor(which_op, sₙ', dag(sₙ))
## end
## 
## function ITensor(o::Op, s::Vector{<:Index})
##   return _ITensor(Tuple(o)..., s)
## end
## 
