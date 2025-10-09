"""
    flux(T::ITensor)

Returns the flux of the ITensor.

If the ITensor is empty or it has no QNs, returns `nothing`.
"""
flux(T::ITensor, args...) = flux(tensor(T), args...)

"""
    checkflux(T::ITensor)

Check that fluxes of all non-zero blocks of a blocked or symmetric ITensor
are equal. Throws an error if one or more blocks have a different flux.
"""
checkflux(T::ITensor, flux_check) = checkflux(tensor(T), flux_check)

"""
    checkflux(T::ITensor, flux)

Check that fluxes of all non-zero blocks of a blocked or symmetric Tensor
equal the value `flux`. Throws an error if one or more blocks does not have this flux.
"""
checkflux(T::ITensor) = checkflux(tensor(T))

#
# Tensor versions
# TODO: Move to NDTensors when QN functionality
# is moved there.
#

"""
    flux(T::Tensor, block::Block)

Compute the flux of a specific block of a Tensor,
regardless of whether this block is present or not in the storage.
"""
flux(T::Tensor, block::Block) = flux(inds(T), block)

"""
    flux(T::Tensor, i::Integer, is::Integer...)

Compute the flux of a specific element of a Tensor,
regardless of whether this element is zero or non-zero.
"""
flux(T::Tensor, i::Integer, is::Integer...) = flux(inds(T), i, is...)

"""
    flux(T::Tensor)

Return the flux of a Tensor, based on what non-zero blocks it
has. If the Tensor is not blocked or has no non-zero blocks,
this function returns `nothing`.
"""
function flux(T::Tensor)
    (!hasqns(T) || isempty(T)) && return nothing
    @debug_check checkflux(T)
    block1 = first(eachnzblock(T))
    return flux(T, block1)
end

allfluxequal(T::Tensor, flux_to_check) = all(b -> flux(T, b) == flux_to_check, nzblocks(T))
allfluxequal(T::Tensor) = allequal(flux(T, b) for b in nzblocks(T))

"""
    checkflux(T::Tensor)

Check that fluxes of all non-zero blocks of a blocked or symmetric Tensor
are equal. Throws an error if one or more blocks have a different flux.
If the tensor is dense (is not blocked) then `checkflux` returns `nothing`.
"""
function checkflux(T::Tensor)
    (!hasqns(T) || isempty(T)) && return nothing
    return allfluxequal(T) ? nothing : error("Fluxes not all equal")
end

"""
    checkflux(T::Tensor, flux)

Check that fluxes of all non-zero blocks of a blocked or symmetric Tensor
equal the value `flux`. Throws an error if one or more blocks does not have this flux.
"""
checkflux(T::Tensor, flux) = allfluxequal(T, flux) ? nothing : error("Fluxes not all equal")
