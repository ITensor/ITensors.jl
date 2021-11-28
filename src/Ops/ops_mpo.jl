function SiteOp(o::Op)
  return SiteOp(Ops.which_op(o), Ops.sites(o), Ops.params(o))
end

function MPOTerm(o::α{∏{Op}})
  return MPOTerm(coefficient(o), [SiteOp(oₙ) for oₙ in o])
end

function OpSum(o::∑{<:α{∏{Op}}})
  return OpSum([MPOTerm(oₙ) for oₙ in o])
end

function OpSum(o::Union{Op,Applied})
  return OpSum(∑{<:α{∏{Op}}}(o))
end

# Conversions from other formats
function MPO(o::Union{Op,Applied}, s::Vector{<:Index}; kwargs...)
  return MPO(OpSum(o), s; kwargs...)
end
