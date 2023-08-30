flux(T::ITensor, args...) = flux(tensor(T), args...)

"""
    flux(T::ITensor)

Returns the flux of the ITensor.

If the ITensor is empty or it has no QNs, returns `nothing`.
"""
function flux(T::ITensor)
  return flux(tensor(T))
end

function checkflux(T::ITensor, flux_check)
  return checkflux(tensor(T), flux_check)
end

function checkflux(T::ITensor)
  return checkflux(tensor(T))
end

#
# Tensor versions
# TODO: Move to NDTensors when QN functionality
# is moved there.
#

flux(T::Tensor, args...) = flux(inds(T), args...)

function flux(T::Tensor)
  (!hasqns(T) || isempty(T)) && return nothing
  @debug_check checkflux(T)
  block1 = first(eachnzblock(T))
  return flux(T, block1)
end

function checkflux(T::Tensor, flux_check)
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

function checkflux(T::Tensor)
  b1 = first(nzblocks(T))
  fluxTb1 = flux(T, b1)
  return checkflux(T, fluxTb1)
end
