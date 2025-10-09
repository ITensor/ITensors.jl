# Fix for AD
function _tr(T::ITensor; plev::Pair{Int, Int} = 0 => 1, tags::Pair = ts"" => ts"")
    trpairs = indpairs(T; plev = plev, tags = tags)
    Cᴸ = combiner(first.(trpairs))
    Cᴿ = combiner(last.(trpairs))
    Tᶜ = T * Cᴸ * Cᴿ
    cᴸ = uniqueind(Cᴸ, T)
    cᴿ = uniqueind(Cᴿ, T)
    Tᶜ *= δ(eltype(T), dag((cᴸ, cᴿ)))
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
function exp(A::ITensor, Linds, Rinds; ishermitian = false)
    @debug_check begin
        if hasqns(A)
            @assert flux(A) == QN()
        end
    end

    NL = length(Linds)
    NR = length(Rinds)
    NL != NR && error("Must have equal number of left and right indices")
    ndims(A) != NL + NR &&
        error("Number of left and right indices must add up to total number of indices")

    # Replace Linds and Rinds with index sets having
    # same id's but arrow directions as on A
    Linds = permute(commoninds(A, Linds), Linds)
    Rinds = permute(commoninds(A, Rinds), Rinds)

    # Ensure indices have correct directions, QNs, etc.
    for (l, r) in zip(Linds, Rinds)
        if space(l) != space(r)
            error("In exp, indices must come in pairs with equal spaces.")
        end
        if hasqns(A) && dir(l) == dir(r)
            error("In exp, indices must come in pairs with opposite directions")
        end
    end

    # <fermions>
    fermionic_itensor = using_auto_fermion() && has_fermionic_subspaces(inds(A))
    if fermionic_itensor
        # If fermionic, bring indices into i',j',..,dag(j),dag(i)
        # ordering with Out indices coming before In indices
        # Resulting tensor acts like a normal matrix (no extra signs
        # when taking powers A^n)
        if all(j -> dir(j) == Out, Linds) && all(j -> dir(j) == In, Rinds)
            ordered_inds = [Linds..., reverse(Rinds)...]
        elseif all(j -> dir(j) == Out, Rinds) && all(j -> dir(j) == In, Linds)
            ordered_inds = [Rinds..., reverse(Linds)...]
        else
            error(
                "For fermionic exp, Linds and Rinds must have same directions within each set. Got dir.(Linds)=",
                dir.(Linds),
                ", dir.(Rinds)=",
                dir.(Rinds),
            )
        end
        A = permute(A, ordered_inds)
        # A^n now sign free, ok to temporarily disable fermion system
        disable_auto_fermion()
    end

    CL = combiner(Linds...; dir = Out)
    CR = combiner(Rinds...; dir = In)
    AC = (A * CR) * CL
    expAT = ishermitian ? exp(Hermitian(tensor(AC))) : exp(tensor(AC))
    expA = (itensor(expAT) * dag(CR)) * dag(CL)

    # <fermions>
    if fermionic_itensor
        # Ensure expA indices in "matrix" form before re-enabling fermion system
        expA = permute(expA, ordered_inds)
        enable_auto_fermion()
    end

    return expA
end

function exp(A::ITensor; kwargs...)
    Ris = filterinds(A; plev = 0)
    Lis = dag.(prime.(Ris))
    return exp(A, Lis, Ris; kwargs...)
end

using NDTensors: NDTensors, map_diag, map_diag!
function NDTensors.map_diag!(f::Function, it_destination::ITensor, it_source::ITensor)
    map_diag!(f, tensor(it_destination), tensor(it_source))
    return it_destination
end
NDTensors.map_diag(f::Function, it::ITensor) = itensor(map_diag(f, tensor(it)))
