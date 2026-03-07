using .QuantumNumbers: QN, QuantumNumbers

"""
    parity_sign(P)

Given an array or tuple of integers representing
a permutation or a subset of a permutation,
compute the parity sign defined as -1 for a
permutation consisting of an odd number of swaps
and +1 for an even number of swaps. This
implementation uses an O(n^2) algorithm and is
intended for small permutations only.
"""
function parity_sign(P, L::Int = length(P))::Int
    s = +1
    for i in 1:L, j in (i + 1):L
        s *= sign(P[j] - P[i])
    end
    return s
end

isfermionic(qv::QuantumNumbers.QNVal) = (QuantumNumbers.modulus(qv) < 0)

isfermionic(qn::QN) = any(isfermionic, qn)

has_fermionic_subspaces(i::Index) = false

function has_fermionic_subspaces(i::QNIndex)
    for b in 1:nblocks(i)
        isfermionic(qn(i, b)) && (return true)
    end
    return false
end

isfermionic(i::Index) = has_fermionic_subspaces(i)

has_fermionic_subspaces(is::Indices) = false

function has_fermionic_subspaces(is::QNIndices)
    for i in is, b in 1:nblocks(i)
        isfermionic(qn(i, b)) && (return true)
    end
    return false
end

has_fermionic_subspaces(T::Tensor) = has_fermionic_subspaces(inds(T))

"""
    fparity(qn::QN)
    fparity(qn::IndexVal)

Compute the fermion parity (0 or 1) of a QN of IndexVal,
defined as the sum mod 2 of each of its fermionic
QNVals (QNVals with negative modulus).
"""
function fparity(qn::QN)
    p = 0
    for qv in qn
        if isfermionic(qv)
            p += val(qv)
        end
    end
    return mod(p, 2)
end

fparity(iv::Pair{<:Index}) = fparity(qn(iv))

Base.isodd(q::QN) = isodd(fparity(q))
Base.isodd(iv::Pair{<:Index}) = isodd(fparity(iv))

# Given a permutation `p` and a precomputed tuple of parities (0=even, 1=odd),
# return the sign of `p` restricted to the odd-parity positions: +1 or -1.
@inline function compute_permfactor_from_parities(
        p, parity::NTuple{N, Int}; range = 1:N
    )::Int where {N}
    !using_auto_fermion() && return 1
    s = +1
    @inbounds for ri in range
        parity[p[ri]] == 0 && continue
        @inbounds for rj in (ri + 1):last(range)
            parity[p[rj]] == 0 && continue
            s *= sign(p[rj] - p[ri])
        end
    end
    return s
end

# Given a permutation `p` and a tuple of QNIndexVals or QNs, return the sign
# of `p` restricted to the fermion-parity-odd elements: +1 or -1.
function compute_permfactor_from_qns(
        p, iv_or_qn::NTuple{N}; range = 1:N
    )::Int where {N}
    !using_auto_fermion() && return 1
    parity = ntuple(i -> fparity(iv_or_qn[i]), Val(N))
    return compute_permfactor_from_parities(p, parity; range)
end

# Varargs entry point — collects individual QNs/IndexVals into a tuple.
function compute_permfactor(p, iv_or_qn...; kws...)
    return compute_permfactor_from_qns(p, iv_or_qn; kws...)
end

function NDTensors.permfactor(p, ivs::Vararg{Pair{QNIndex}, N}; kwargs...) where {N}
    !using_auto_fermion() && return 1
    return compute_permfactor_from_qns(p, ivs; kwargs...)
end

function NDTensors.permfactor(
        perm, block::NDTensors.Block{N}, inds::QNIndices; range = 1:N
    ) where {N}
    !using_auto_fermion() && return 1
    parity = ntuple(n -> fparity(qn(inds[n], block[n])), Val(N))
    return compute_permfactor_from_parities(perm, parity; range)
end

NDTensors.block_parity(i::QNIndex, block::Integer) = fparity(qn(i, block))

NDTensors.block_sign(i::QNIndex, block::Integer) = 2 * NDTensors.block_parity(i, block) - 1

function NDTensors.right_arrow_sign(i::QNIndex, block::Integer)
    !using_auto_fermion() && return 1
    if dir(i) == Out && NDTensors.block_parity(i, block) == 1
        return -1
    end
    return 1
end

function NDTensors.left_arrow_sign(i::QNIndex, block::Integer)
    !using_auto_fermion() && return 1
    if dir(i) == In && NDTensors.block_parity(i, block) == 1
        return -1
    end
    return 1
end

# Version of getperm which is type stable
# and works for Tuple or Vector inputs
function vec_getperm(s1, s2)
    N = length(s1)
    p = Vector{Int}(undef, N)
    for i in 1:N
        @inbounds p[i] = NDTensors._findfirst(==(@inbounds s1[i]), s2)
    end
    return p
end

@inline function NDTensors.compute_alpha(
        ElR,
        labelsR,
        blockR::NDTensors.Block{NR},
        input_indsR,
        labelsT1,
        blockT1,
        indsT1::NTuple{N1, QNIndex},
        labelsT2,
        blockT2,
        indsT2::NTuple{N2, QNIndex}
    ) where {N1, N2, NR}
    if !using_auto_fermion()
        !has_fermionic_subspaces(indsT1) || !has_fermionic_subspaces(indsT2)
        return one(ElR)
    end

    # input_indsR is a QNIndices (Tuple{Vararg{QNIndex}}), use it directly.
    indsR = input_indsR

    nlabelsT1 = TupleTools.sort(labelsT1; rev = true)
    nlabelsT2 = TupleTools.sort(labelsT2)

    # Make orig_labelsR from the uncontracted (positive) labels of T1 then T2.
    # After sorting T1 descending, positive labels occupy the first NP1 positions.
    # After sorting T2 ascending, positive labels occupy the last NP2 positions.
    # NP1 and NP2 are compile-time constants (functions of N1, N2, NR).
    # Build orig_labelsR as a pure NTuple to avoid any heap allocation.
    NP1 = (NR + N1 - N2) ÷ 2   # uncontracted labels from T1
    NP2 = NR - NP1               # uncontracted labels from T2
    orig_labelsR = (
        ntuple(i -> nlabelsT1[i], Val(NP1))...,
        ntuple(i -> nlabelsT2[N2 - NP2 + i], Val(NP2))...,
    )

    permT1 = NDTensors.getperm(nlabelsT1, labelsT1)
    permT2 = NDTensors.getperm(nlabelsT2, labelsT2)
    permR = NDTensors.getperm(labelsR, orig_labelsR)

    # Precompute parities once per block — avoids redundant qn lookups across the
    # three permfactor calls and the alpha_arrows loop below (each qn call is ~35 ns).
    parityT1 = ntuple(n -> fparity(qn(indsT1[n], blockT1[n])), Val(N1))
    parityT2 = ntuple(n -> fparity(qn(indsT2[n], blockT2[n])), Val(N2))
    parityR = ntuple(n -> fparity(qn(indsR[n], blockR[n])), Val(NR))

    alpha1 = compute_permfactor_from_parities(permT1, parityT1)
    alpha2 = compute_permfactor_from_parities(permT2, parityT2)
    alphaR = compute_permfactor_from_parities(permR, parityR)

    alpha_arrows = one(ElR)
    @inbounds for n in 1:N1
        labelsT1[n] < 0 && dir(indsT1[n]) == Out && parityT1[n] == 1 && (alpha_arrows *= -1)
    end

    α = alpha1 * alpha2 * alphaR * alpha_arrows

    return α
end

# Flip signs of selected blocks of T prior to
# it being multiplied by a combiner ITensor
# labelsR gives the ordering of indices after the product
function NDTensors.before_combiner_signs(
        T,
        labelsT_,
        indsT::NTuple{NT, QNIndex},
        C,
        labelsC_,
        indsC::NTuple{NC, QNIndex},
        labelsR,
        indsR::NTuple{NR, QNIndex}
    ) where {NC, NT, NR}
    if !using_auto_fermion() || !has_fermionic_subspaces(T)
        return T
    end

    T = copy(T)

    # Convert labels to NTuples — NC and NT are compile-time constants, so these
    # are stack-allocated and avoid the Vector allocation from list comprehensions.
    labelsC = ntuple(i -> labelsC_[i], Val(NC))
    labelsT = ntuple(i -> labelsT_[i], Val(NT))

    ci = NDTensors.cinds(storage(C))[1]
    combining = (labelsC[ci] > 0)

    isconj = NDTensors.isconj(storage(C))

    if combining
        #println("Combining <<<<<<<<<<<<<<<<<<<<<<<<<<<")

        # Uncombined index labels from combiner (positions 2:NC, all negative/contracted).
        # Use MVector to fill nlabelsT without heap allocation (NT is compile-time constant).
        uc_labels = ntuple(i -> labelsC[i + 1], Val(NC - 1))
        if isconj
            uc_labels = reverse(uc_labels)
        end
        @assert all(l -> l < 0, uc_labels)

        nlabelsT = MVector{NT, Int}(undef)
        u = 1
        for l in uc_labels
            nlabelsT[u] = l
            u += 1
        end
        for l in labelsT
            if l > 0 #uncontracted
                nlabelsT[u] = l
                u += 1
            end
        end
        @assert u == NT + 1

        # Compute permutation as NTuple (stack-allocated; Val(NT) is compile-time constant).
        permT = ntuple(i -> NDTensors._findfirst(==(nlabelsT[i]), labelsT), Val(NT))

        for blockT in keys(blockoffsets(T))
            # Compute sign from permuting uncombined indices to front:
            alphaT = NDTensors.permfactor(permT, blockT, indsT)

            neg_dir = !isconj ? In : Out
            alpha_arrows = 1
            alpha_mixed_arrow = 1
            C_dir = dir(indsC[1])
            for n in 1:NT
                i = indsT[n]
                qi = qn(i, blockT[n])
                if labelsT[n] < 0 && fparity(qi) == 1
                    alpha_mixed_arrow *= (dir(i) != C_dir) ? -1 : +1
                    alpha_arrows *= (dir(i) == neg_dir) ? -1 : +1
                end
            end

            fac = alphaT * alpha_arrows

            if isconj
                fac *= alpha_mixed_arrow
            end

            if fac != 1
                Tb = blockview(T, blockT)
                scale!(Tb, fac)
            end
        end # for blockT

    elseif !combining
        #
        # Uncombining ---------------------------
        #
        #println("Uncombining >>>>>>>>>>>>>>>>>>>>>>>>>>>")

        nc = 0
        for n in 1:NT
            if labelsT[n] < 0
                nc = n
                break
            end
        end
        ic = indsT[nc]

        nlabelsT = MVector{NT, Int}(undef)
        nlabelsT[1] = labelsT[nc]
        u = 2
        for l in labelsT
            if l > 0
                nlabelsT[u] = l
                u += 1
            end
        end

        # Compute sign for permuting combined index to front
        # (sign alphaT to be computed for each block below):
        permT = ntuple(i -> NDTensors._findfirst(==(nlabelsT[i]), labelsT), Val(NT))

        #
        # Note: other permutation of labelsT which
        # relates to two treatments of isconj==true/false
        # in combining case above is handled as a
        # post-processing step in NDTensors.after_combiner_signs
        # implemented below
        #

        for blockT in keys(blockoffsets(T))
            alphaT = NDTensors.permfactor(permT, blockT, indsT)

            neg_dir = !isconj ? Out : In
            qic = qn(ic, blockT[nc])
            alpha_arrows = (fparity(qic) == 1 && dir(ic) == neg_dir) ? -1 : +1

            fac = alphaT * alpha_arrows

            if fac != 1
                Tb = blockview(T, blockT)
                scale!(Tb, fac)
            end
        end
    end

    return T
end

function NDTensors.after_combiner_signs(
        R, labelsR, indsR::NTuple{NR, QNIndex}, C, labelsC, indsC::NTuple{NC, QNIndex}
    ) where {NC, NR}
    ci = NDTensors.cinds(store(C))[1]
    combining = (labelsC[ci] > 0)
    combining && error("NDTensors.after_combiner_signs only for uncombining")

    if !using_auto_fermion() || !has_fermionic_subspaces(R)
        return R
    end

    R = copy(R)

    # number of uncombined indices
    Nuc = NC - 1

    isconj = NDTensors.isconj(store(C))

    if !combining
        if !isconj
            #println("!!! Doing uncombining post-processing step")
            rperm = ntuple(i -> (Nuc - i + 1), Nuc) # reverse permutation
            NDTensors.scale_blocks!(
                R, block -> NDTensors.permfactor(rperm, block, indsR; range = 1:Nuc)
            )
        else
            #println("!!! Doing conjugate uncombining post-processing step")
            C_dir = dir(inds(C)[1])

            function mixed_arrow_sign(block)
                alpha_mixed_arrow = 1
                for n in 1:Nuc
                    i = indsR[n]
                    qi = qn(i, block[n])
                    if dir(i) == C_dir && fparity(qi) == 1
                        alpha_mixed_arrow *= -1
                    end
                end
                return alpha_mixed_arrow
            end

            NDTensors.scale_blocks!(R, block -> mixed_arrow_sign(block))
        end
    end

    return R
end
