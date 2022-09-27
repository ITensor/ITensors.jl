# Fix for AD
function _tr(T::ITensor; plev::Pair{Int,Int}=0 => 1, tags::Pair=ts"" => ts"")
  trpairs = indpairs(T; plev=plev, tags=tags)
  Cᴸ = combiner(first.(trpairs))
  Cᴿ = combiner(last.(trpairs))
  Tᶜ = T * Cᴸ * Cᴿ
  cᴸ = uniqueind(Cᴸ, T)
  cᴿ = uniqueind(Cᴿ, T)
  Tᶜ *= δ(dag((cᴸ, cᴿ)))
  if order(Tᶜ) == 0
    return Tᶜ[]
  end
  return Tᶜ
end

# Trace an ITensor over pairs of indices determined by
# the prime levels and tags. Indices that are not in pairs
# are not traced over, corresponding to a "batched" trace.
function tr(T::ITensor; kwargs...)
  return _tr(T; kwargs...)
end

"""
    exp(A::ITensor, Linds=Rinds', Rinds=inds(A,plev=0); ishermitian = false)

Compute the exponential of the tensor `A` by treating it as a matrix ``A_{lr}`` with
the left index `l` running over all indices in `Linds` and `r` running over all
indices in `Rinds`.

Only accepts index lists `Linds`,`Rinds` such that: (1) `length(Linds) +
length(Rinds) == length(inds(A))` (2) `length(Linds) == length(Rinds)` (3) For
each pair of indices `(Linds[n],Rinds[n])`, `Linds[n]` and `Rinds[n]` represent
the same Hilbert space (the same QN structure in the QN case, or just the same
length in the dense case), and appear in `A` with opposite directions.

When `ishermitian=true` the exponential of `Hermitian(A_{lr})` is
computed internally.
"""
function exp(A::ITensor, Linds, Rinds; kwargs...)
  ishermitian = get(kwargs, :ishermitian, false)

  @debug_check begin
    if hasqns(A)
      @assert flux(A) == QN()
    end
  end

  N = ndims(A)
  NL = length(Linds)
  NR = length(Rinds)
  NL != NR && error("Must have equal number of left and right indices")
  N != NL + NR &&
    error("Number of left and right indices must add up to total number of indices")

  # Linds, Rinds may not have the correct directions
  # TODO: does the need a conversion?
  Lis = Linds
  Ris = Rinds

  # Ensure the indices have the correct directions,
  # QNs, etc.
  # First grab the indices in A, then permute them
  # correctly.
  Lis = permute(commoninds(A, Lis), Lis)
  Ris = permute(commoninds(A, Ris), Ris)

  for (l, r) in zip(Lis, Ris)
    if space(l) != space(r)
      error("In exp, indices must come in pairs with equal spaces.")
    end
    if hasqns(A)
      if dir(l) == dir(r)
        error("In exp, indices must come in pairs with opposite directions")
      end
    end
  end

  CL = combiner(Lis...; dir=Out)
  CR = combiner(Ris...; dir=In)
  AC = (A * CR) * CL
  expAT = ishermitian ? exp(Hermitian(tensor(AC))) : exp(tensor(AC))
  return (itensor(expAT) * dag(CR)) * dag(CL)
end

function exp(A::ITensor; kwargs...)
  Ris = filterinds(A; plev=0)
  Lis = Ris'
  return exp(A, Lis, Ris; kwargs...)
end
