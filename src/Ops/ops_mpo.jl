function SiteOp(o::Op)
  return SiteOp(Ops.which_op(o), Ops.sites(o), Ops.params(o))
end

function MPOTerm(o::Ops.ScaledProdOp)
  return MPOTerm(coefficient(o), [SiteOp(oₙ) for oₙ in o])
end

function OpSum(o::Ops.OpSum)
  return OpSum([MPOTerm(oₙ) for oₙ in o])
end

function MPO(o::Ops.OpSum, s::Vector{<:Index}; kwargs...)
  return MPO(OpSum(o), s; kwargs...)
end

# Conversions from other formats
function MPO(o::Union{Op,Applied}, s::Vector{<:Index}; kwargs...)
  return MPO(convert(Ops.SumScaledProdOp, o), s; kwargs...)
end
