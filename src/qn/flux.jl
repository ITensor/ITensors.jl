flux(T::Union{Tensor,ITensor}, args...) = flux(inds(T), args...)

"""
    flux(T::ITensor)

Returns the flux of the ITensor.

If the ITensor is empty or it has no QNs, returns `nothing`.
"""
function flux(T::Union{Tensor,ITensor})
  (!hasqns(T) || isempty(T)) && return nothing
  @debug_check checkflux(T)
  block1 = first(eachnzblock(T))
  return flux(T, block1)
end

function checkflux(T::Union{Tensor,ITensor}, flux_check)
  for b in nzblocks(T)
    fluxTb = flux(T, b)
    if fluxTb != flux_check
      error(
        "Block $b has flux $fluxTb that is inconsistent with the desired flux $flux_check"
      )
    end
  end
  return nothing
end

function checkflux(T::Union{Tensor,ITensor})
  b1 = first(nzblocks(T))
  fluxTb1 = flux(T, b1)
  return checkflux(T, fluxTb1)
end
