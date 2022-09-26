## function apply(o::Prod{ITensor}, v::Union{MPS,MPO}; kwargs...)
##   ov = v
##   for oₙ in only(o.args)
##     ov = apply(oₙ, ov; kwargs...)
##   end
##   return ov
## end
## 
## function (o::Prod{ITensor})(v::Union{MPS,MPO}; kwargs...)
##   return apply(o, v; kwargs...)
## end

#
# Conversion to ITensors.OpSum and MPO
#

## function SiteOp(o::Op)
##   return SiteOp(Ops.which_op(o), Ops.sites(o), Ops.params(o))
## end
## 
## function MPOTerm(o::Scaled{C,Prod{Op}}) where {C}
##   return MPOTerm(coefficient(o), [SiteOp(oₙ) for oₙ in argument(o)])
## end
## 
## function OpSum(o::Sum{Scaled{C,Prod{Op}}}) where {C}
##   return OpSum([MPOTerm(oₙ) for oₙ in o])
## end

## function OpSum(o::Union{Op,Applied})
##   return OpSum(Sum{<:Scaled{<:Number,Prod{Op}}}(o))
## end

## function OpSum(
##   o::Union{Op,Scaled{C,Op},Prod{Op},Sum{Op},Scaled{C,Prod{Op}},Sum{Scaled{Float64,Op}}}
## ) where {C}
##   os = Sum{Scaled{Float64,Prod{Op}}}() + o
##   return OpSum(os)
## end
## 
## # Conversions from other formats
## function MPO(o::Union{Op,Applied}, s::Vector{<:Index}; kwargs...)
##   return MPO(OpSum(o), s; kwargs...)
## end
