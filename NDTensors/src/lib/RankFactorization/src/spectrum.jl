"""
  Spectrum
contains the (truncated) density matrix eigenvalue spectrum which is computed during a
decomposition done by `svd` or `eigen`. In addition stores the truncation error.
"""
struct Spectrum{VecT <: Union{AbstractVector, Nothing}, ElT <: Real}
    eigs::VecT
    truncerr::ElT
end

eigs(s::Spectrum) = s.eigs
truncerror(s::Spectrum) = s.truncerr

function entropy(s::Spectrum)
    S = 0.0
    eigs_s = eigs(s)
    isnothing(eigs_s) &&
        error("Spectrum does not contain any eigenvalues, cannot compute the entropy")
    for p in eigs_s
        p > 1.0e-13 && (S -= p * log(p))
    end
    return S
end
