"""
Some of the functionality in this script is closely related to routines in the following package
https://github.com/Jacupo/F_utilities
and the associated publication
10.21468/SciPostPhysLectNotes.54
"""

"""Takes a single-particle Hamiltonian in blocked Dirac format and finds the fermionic transformation U that diagonalizes it"""
function _eigen_gaussian_blocked(H; noise_scale=nothing)
  #make sure H is Hermitian
  @assert ishermitian(H)
  H = Hermitian(H)
  ElT = eltype(H)
  #convert from Dirac to Majorana picture
  N = size(H, 1)
  Ω = build_Ω(ElT, N)
  h = real(-im .* (Ω * H * Ω'))
  h = (h - h') ./ 2
  #@show size(h)
  if !isnothing(noise_scale)
    noise = rand(size(h)...) * noise_scale
    noise = (noise - noise') ./ 2
    h = h + noise
  end
  # Schur diagonalize including reordering
  _, O, vals = order_schur(schur(h))
  # convert back to Dirac Frame
  Fxpxx = build_Fxpxx(N)
  U = Ω' * O * (Fxpxx') * Ω
  d = vcat(-vals, vals)
  if ElT <: Real
    U = make_real_if_possible(U, d)
    # make another pass with rotation in the complex plane per eigenvector
    U .*= exp.(-im * angle.(U[1:1, :]))
    @assert norm(imag.(U)) < sqrt(eps(real(ElT)))
    U = real(real.(U))
  end
  return d, U
end

"""Takes a single-particle Hamiltonian in interlaced Dirac format and finds the complex fermionic transformation U that diagonalizes it."""
function eigen_gaussian(H; noise_scale=nothing)
  d, U = _eigen_gaussian_blocked(
    ITensorGaussianMPS.reverse_interleave(complex(H)); noise_scale=noise_scale
  )
  nU = similar(U)
  n = div(size(H, 1), 2)
  nU[1:2:end, :] = U[1:n, :]
  nU[2:2:end, :] = U[(n + 1):end, :]
  return d, nU
end

"""Takes a single-particle Hamiltonian in interlaced Dirac format and outputs the ground state correlation matrix (with the input Hamiltonians element type)."""
function get_gaussian_GS_corr(H::AbstractMatrix; noise_scale=nothing)
  ElT = eltype(H)
  d, U = eigen_gaussian(H; noise_scale=noise_scale)
  n = div(size(H, 1), 2)
  c = conj(U[:, 1:n]) * transpose(U[:, 1:n])
  if ElT <: Real && norm(imag.(c)) <= sqrt(eps(real(ElT)))
    c = real(real.(c))
  end
  return c
end

"""Takes a single-particle correlation matrix in interlaced Dirac format and finds the fermionic transformation U that diagonalizes it"""
function diag_corr_gaussian(Λ::Hermitian; noise_scale=nothing)
  #shift correlation matrix by half so spectrum is symmetric around 0
  populations, U = eigen_gaussian(Λ - 0.5 * I; noise_scale=noise_scale)
  n = diag(U' * Λ * U)
  if !all(abs.(populations - (n - 0.5 * ones(size(n)))) .< sqrt(eps(real(eltype(Λ)))))
    @show n
    @show populations .+ 0.5
    @error(
      "The natural orbital populations are not consistent, see above. Try adding symmetric noise to the input matrix."
    )
  end
  return populations .+ 0.5, U
end

"""Takes a single-particle correlation matrix in interlaced Dirac format and finds the fermionic transformation U that diagonalizes it"""
function diag_corr_gaussian(Γ::AbstractMatrix; noise_scale=nothing)
  #enforcing hermitianity
  Γ = (Γ + Γ') / 2.0
  return diag_corr_gaussian(Hermitian(Γ); noise_scale=noise_scale)
end

"""Schur decomposition of skew-hermitian matrix"""
function order_schur(F::LinearAlgebra.Schur)
  T = F.Schur
  O = F.vectors #column vectors are Schur vectors

  N = size(T, 1)
  n = div(N, 2)
  shuffled_inds = Vector{Int}[]
  ElT = eltype(T)
  vals = ElT[]
  # build a permutation matrix that takes care of the ordering
  for i in 1:n
    ind = 2 * i - 1
    val = T[ind, ind + 1]
    if real(val) >= 0
      push!(shuffled_inds, [ind, ind + 1])
    else
      push!(shuffled_inds, [ind + 1, ind])
    end
    push!(vals, abs(val))
  end
  # build block local rotation first
  perm = sortperm(real.(vals); rev=true)    ##we want the upper left corner to be the largest absolute value eigval pair?
  vals = vals[perm]
  shuffled_inds = reduce(vcat, shuffled_inds[perm])
  # then permute blocks for overall ordering
  T = T[shuffled_inds, shuffled_inds]
  O = O[:, shuffled_inds]
  return T, O, vals #vals are only positive, and of length n and not N
end

"""Checks if we can make degenerate subspaces of a U0 real by multiplying columns or rows with a phase"""
function make_real_if_possible(U0::AbstractMatrix, spectrum::Vector; sigdigits=12)
  # only apply to first half of spectrum due to symmetry around 0
  # assumes spectrum symmetric around zero and ordered as vcat(-E,E) where E is ordered in descending magnitude
  U = copy(U0)
  n = div(length(spectrum), 2)
  # Round spectrum for comparison within finite floating point precision.
  # Not the cleanest way to compare floating point numbers for approximate equality but should be sufficient here.
  rounded_halfspectrum = round.(spectrum[1:n], sigdigits=sigdigits)
  approx_unique_eigvals = unique(rounded_halfspectrum)
  # loop over degenerate subspaces
  for e in approx_unique_eigvals
    mask = rounded_halfspectrum .== e
    if abs(e) < eps(real(eltype(U0)))
      # handle values close to zero separately
      # rotate subspace for both positive and negative eigenvalue if they are close enough to zero
      mask = vcat(mask, mask)
      subspace = U[:, mask]
      subspace = make_subspace_real_if_possible(subspace)
      U[:, mask] = subspace

    else
      mask = rounded_halfspectrum .== e
      # rotate suspace for the negative eigenvalue
      subspace = U[:, 1:n][:, mask]
      subspace = make_subspace_real_if_possible(subspace)
      v = @views U[:, 1:n][:, mask]
      v .= subspace
      # rotate suspace for the positive eigenvalue
      subspace = U[:, (n + 1):end][:, mask]
      subspace = make_subspace_real_if_possible(subspace)
      v = @views U[:, (n + 1):end][:, mask]
      v .= subspace
    end
  end
  return U
end

"""Checks if we can make a degenerate subspace of the eigenbasis of an operator real by multiplying columns or rows with a phase"""
function make_subspace_real_if_possible(U::AbstractMatrix; atol=sqrt(eps(real(eltype(U)))))
  if eltype(U) <: Real
    return U
  end
  if size(U, 2) == 1
    nU = U .* exp(-im * angle(U[1, 1]))
    if norm(imag.(nU)) .<= atol
      return nU
    else
      return U
    end
  else
    n = size(U, 2)
    gram = U * U'
    if norm(imag.(gram)) .<= atol
      D, V = eigen(Hermitian(real.(gram)))
      proj = V[:, (size(U, 1) - n + 1):end] * (V[:, (size(U, 1) - n + 1):end])'
      @assert norm(proj * U - U) < atol
      return complex(V[:, (size(U, 1) - n + 1):end])
    else
      return U
    end
  end
end

# transformation matrices (in principle sparse) between and within Majorana and Dirac picture

function build_Ω(T, N::Int)
  n = div(N, 2)
  nElT = T <: Real ? Complex{T} : T
  Ω = zeros(nElT, N, N)
  Ω[1:n, 1:n] .= diagm(ones(nElT, n) ./ sqrt(2))
  Ω[1:n, (n + 1):N] .= diagm(ones(nElT, n) ./ sqrt(2))
  Ω[(n + 1):N, 1:n] .= diagm(ones(nElT, n) * (im / sqrt(2)))
  Ω[(n + 1):N, (n + 1):N] .= diagm(ones(nElT, n) * (-im / sqrt(2)))
  return Ω
end

function build_Fxpxx(N::Int)
  Fxpxx = zeros(Int8, N, N)
  n = div(N, 2)
  Fxpxx[1:n, 1:2:N] .= diagm(ones(Int8, n))
  Fxpxx[(n + 1):N, 2:2:N] .= diagm(ones(Int8, n))
  return Fxpxx
end
