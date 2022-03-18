#
# Single particle von Neumann entanglement entropy
#

function entropy(n::Number)
  (n ≤ 0 || n ≥ 1) && return 0
  return -(n * log(n) + (1 - n) * log(1 - n))
end

entropy(ns::Vector{Float64}) = sum(entropy, ns)

#
# Linear Algebra tools
#

"""
    frobenius_distance(M1::AbstractMatrix, M2::AbstractMatrix)

Computes the Frobenius distance `√tr((M1-M2)'*(M1-M2))`.
"""
function frobenius_distance(M1::AbstractMatrix, M2::AbstractMatrix)
  return sqrt(abs(tr(M1'M1) + tr(M2'M2) - tr(M1'M2) - tr(M2'M1)))
end

#
# Rotations
#

struct Circuit{T} <: LinearAlgebra.AbstractRotation{T}
  rotations::Vector{Givens{T}}
end

Base.adjoint(R::Circuit) = Adjoint(R)

function Base.show(io::IO, ::MIME"text/plain", C::Circuit{T}) where {T}
  print(io, "Circuit{$T}:\n")
  return show(io, "text/plain", C.rotations)
end

function Base.copy(aR::Adjoint{<:Any,Circuit{T}}) where {T}
  return Circuit{T}(reverse!([r' for r in aR.parent.rotations]))
end

function LinearAlgebra.lmul!(G::Givens, R::Circuit)
  push!(R.rotations, G)
  return R
end

function LinearAlgebra.lmul!(R::Circuit, A::AbstractArray)
  @inbounds for i in 1:length(R.rotations)
    lmul!(R.rotations[i], A)
  end
  return A
end

function LinearAlgebra.rmul!(A::AbstractMatrix, adjR::Adjoint{<:Any,<:Circuit})
  R = adjR.parent
  @inbounds for i in 1:length(R.rotations)
    rmul!(A, adjoint(R.rotations[i]))
  end
  return A
end

Base.:*(g1::Circuit, g2::Circuit) = Circuit(vcat(g2.rotations, g1.rotations))
LinearAlgebra.lmul!(g1::Circuit, g2::Circuit) = append!(g2.rotations, g1.rotations)

Base.:*(A::Circuit, B::Union{<:Hermitian,<:Diagonal}) = A * convert(Matrix, B)
Base.:*(A::Adjoint{<:Any,<:Circuit}, B::Hermitian) = copy(A) * convert(Matrix, B)
Base.:*(A::Adjoint{<:Any,<:Circuit}, B::Diagonal) = copy(A) * convert(Matrix, B)
function Base.:*(A::Adjoint{<:Any,<:AbstractVector}, B::Adjoint{<:Any,<:Circuit})
  return convert(Matrix, A) * B
end

function LinearAlgebra.rmul!(A::AbstractMatrix, R::Circuit)
  @inbounds for i in reverse(1:length(R.rotations))
    rmul!(A, R.rotations[i])
  end
  return A
end

function shift!(G::Circuit, i::Int)
  for (n, g) in enumerate(G.rotations)
    G.rotations[n] = Givens(g.i1 + i, g.i2 + i, g.c, g.s)
  end
  return G
end

ngates(G::Circuit) = length(G.rotations)

#
# Free fermion tools
#

# Make a hopping Hamiltonian from quadratic Hamiltonian
function hopping_hamiltonian(ampo::AutoMPO)
  nterms = length(ampo.data)
  coefs = Vector{Number}(undef, nterms)
  sites = Vector{Tuple{Int,Int}}(undef, nterms)
  nsites = 0
  for n in 1:nterms
    term = ampo.data[n]
    coef = isreal(term.coef) ? real(term.coef) : term.coef
    coefs[n] = coef
    ops = term.ops
    length(ops) != 2 && error("Must create hopping Hamiltonian from quadratic Hamiltonian")
    sites[n] = ntuple(n -> only(ops[n].site), Val(2))
    nsites = max(nsites, maximum(sites[n]))
  end
  ElT = all(isreal(coefs)) ? Float64 : ComplexF64
  h = zeros(ElT, nsites, nsites)
  for n in 1:nterms
    h[sites[n]...] = coefs[n]
  end
  return h
end

# Make a combined hopping Hamiltonian for spin up and down
function hopping_hamiltonian(ampo_up::AutoMPO, ampo_dn::AutoMPO)
  h_up = hopping_hamiltonian(ampo_up)
  h_dn = hopping_hamiltonian(ampo_dn)
  @assert size(h_up) == size(h_dn)
  N = size(h_up, 1)
  ElT = promote_type(eltype(h_up), eltype(h_dn))
  h = zeros(ElT, 2 * N, 2 * N)
  for i in 1:(2 * N), j in 1:(2 * N)
    if isodd(i) && isodd(j)
      i_up, j_up = (i + 1) ÷ 2, (j + 1) ÷ 2
      h[i, j] = h_up[i_up, j_up]
    elseif iseven(i) && iseven(j)
      i_dn, j_dn = i ÷ 2, j ÷ 2
      h[i, j] = h_dn[i_dn, j_dn]
    end
  end
  return Hermitian(h)
end

# Make a Slater determinant matrix from a hopping Hamiltonian
# h with Nf fermions.
function slater_determinant_matrix(h::AbstractMatrix, Nf::Int)
  _, u = eigen(h)
  return u[:, 1:Nf]
end

#
# Correlation matrix diagonalization
#

"""
    givens_rotations(v::AbstractVector)

For a vector `v`, return the `length(v)-1`
Givens rotations `g` and the norm `r` such that:
```julia
g * v ≈ r * [n == 1 ? 1 : 0 for n in 1:length(v)]
```
"""
function givens_rotations(v::AbstractVector{ElT}) where {ElT}
  N = length(v)
  gs = Circuit{ElT}([])
  r = v[1]
  for n in reverse(1:(N - 1))
    g, r = givens(v, n, n + 1)
    v = g * v
    lmul!(g, gs)
  end
  return gs, r
end

"""
    correlation_matrix_to_gmps(Λ::AbstractMatrix{ElT}; eigval_cutoff::Float64 = 1e-8, maxblocksize::Int = size(Λ0, 1))

Diagonalize a correlation matrix, returning the eigenvalues and eigenvectors
stored in a structure as a set of Givens rotations.

The correlation matrix should be Hermitian, and will be treated as if it itensor
in the algorithm.
"""
function correlation_matrix_to_gmps(
  Λ0::AbstractMatrix{ElT}; eigval_cutoff::Float64=1e-8, maxblocksize::Int=size(Λ0, 1)
) where {ElT<:Number}
  Λ = Hermitian(Λ0)
  N = size(Λ, 1)
  V = Circuit{ElT}([])
  ns = Vector{real(ElT)}(undef, N)
  err_tot = 0.0
  for i in 1:N
    blocksize = 0
    n = 0.0
    err = 0.0
    p = Int[]
    uB = 0.0
    for blocksize in 1:maxblocksize
      j = min(i + blocksize, N)
      ΛB = @view Λ[i:j, i:j]
      nB, uB = eigen(Hermitian(ΛB))
      p = sortperm(nB; by=entropy)
      n = nB[p[1]]
      err = min(n, 1 - n)
      err ≤ eigval_cutoff && break
    end
    err_tot += err
    ns[i] = n
    v = @view uB[:, p[1]]
    g, _ = givens_rotations(v)
    shift!(g, i - 1)
    # In-place version of:
    # V = g * V
    lmul!(g, V)
    Λ = Hermitian(g * Λ * g')
  end
  return ns, V
end

function slater_determinant_to_gmps(Φ::AbstractMatrix; kwargs...)
  return correlation_matrix_to_gmps(conj(Φ) * transpose(Φ); kwargs...)
end

#
# Turn circuit into MPS
#

function ITensors.ITensor(u::Givens, s1::Index, s2::Index)
  U = [
    1 0 0 0
    0 u.c u.s 0
    0 -conj(u.s) u.c 0
    0 0 0 1
  ]
  return itensor(U, s2', s1', dag(s2), dag(s1))
end

function ITensors.ITensor(sites::Vector{<:Index}, u::Givens)
  s1 = sites[u.i1]
  s2 = sites[u.i2]
  return ITensor(u, s1, s2)
end

"""
    MPS(sites::Vector{<:Index}, state, U::Vector{<:ITensor}; kwargs...)

Return an MPS with site indices `sites` by applying the circuit `U` to the starting state `state`.
"""
function ITensors.MPS(sites::Vector{<:Index}, state, U::Vector{<:ITensor}; kwargs...)
  return apply(U, productMPS(sites, state); kwargs...)
end

function isspinful(s::Index)
  !hasqns(s) && return false
  return all(qnblock -> ITensors.hasname(qn(qnblock), ITensors.QNVal("Sz", 0)), space(s))
end

function isspinful(s::Vector{<:Index})
  return all(isspinful, s)
end

"""
    correlation_matrix_to_mps(s::Vector{<:Index}, Λ::AbstractMatrix{ElT};
                              eigval_cutoff::Float64 = 1e-8,
                              maxblocksize::Int = size(Λ, 1),
                              kwargs...)

Return an approximation to the state represented by the correlation matrix as
a matrix product state (MPS).

The correlation matrix should correspond to a pure state (have all eigenvalues
of zero or one).
"""
function correlation_matrix_to_mps(
  s::Vector{<:Index},
  Λ::AbstractMatrix;
  eigval_cutoff::Float64=1e-8,
  maxblocksize::Int=size(Λ, 1),
  kwargs...,
)
  @assert size(Λ, 1) == size(Λ, 2)
  ns, C = correlation_matrix_to_gmps(
    Λ; eigval_cutoff=eigval_cutoff, maxblocksize=maxblocksize
  )
  if all(hastags("Fermion"), s)
    U = [ITensor(s, g) for g in reverse(C.rotations)]
    ψ = MPS(s, n -> round(Int, ns[n]) + 1, U; kwargs...)
  elseif all(hastags("Electron"), s)
    isodd(length(s)) && error(
      "For Electron type, must have even number of sites of alternating up and down spins.",
    )
    N = length(s)
    if isspinful(s)
      error(
        "correlation_matrix_to_mps(Λ::AbstractMatrix) currently only supports spinless Fermions or Electrons that do not conserve Sz. Use correlation_matrix_to_mps(Λ_up::AbstractMatrix, Λ_dn::AbstractMatrix) to use spinful Fermions/Electrons.",
      )
    else
      sf = siteinds("Fermion", 2 * N; conserve_qns=true)
    end
    U = [ITensor(sf, g) for g in reverse(C.rotations)]
    ψf = MPS(sf, n -> round(Int, ns[n]) + 1, U; kwargs...)
    ψ = MPS(N)
    for n in 1:N
      i, j = 2 * n - 1, 2 * n
      C = combiner(sf[i], sf[j])
      c = combinedind(C)
      ψ[n] = ψf[i] * ψf[j] * C
      ψ[n] *= δ(dag(c), s[n])
    end
  else
    error("All sites must be Fermion or Electron type.")
  end
  return ψ
end

function slater_determinant_to_mps(s::Vector{<:Index}, Φ::AbstractMatrix; kwargs...)
  return correlation_matrix_to_mps(s, conj(Φ) * transpose(Φ); kwargs...)
end

function slater_determinant_to_mps(
  s::Vector{<:Index}, Φ_up::AbstractMatrix, Φ_dn::AbstractMatrix; kwargs...
)
  return correlation_matrix_to_mps(
    s, conj(Φ_up) * transpose(Φ_up), conj(Φ_dn) * transpose(Φ_dn); kwargs...
  )
end

function mapindex(f::Function, C::Circuit)
  return Circuit(mapindex.(f, C.rotations))
end

function mapindex(f::Function, g::Givens)
  return Givens(f(g.i1), f(g.i2), g.c, g.s)
end

function identity_blocks!(T::Tensor)
  for b in nzblocks(T)
    T[b] = Matrix{Float64}(I, dims(T[b]))
  end
  return T
end

# Creates an ITensor with the specified flux where each nonzero block
# is identity
# TODO: make a special constructor for this.
function identity_blocks_itensor(flux::QN, i1::Index, i2::Index)
  A = ITensor(flux, i1, i2)
  identity_blocks!(tensor(A))
  return A
end

function identity_blocks_itensor(i1::ITensors.QNIndex, i2::ITensors.QNIndex)
  return identity_blocks_itensor(QN(), i1, i2)
end

function identity_blocks_itensor(i1::Index, i2::Index)
  M = Matrix{Float64}(I, dim(i1), dim(i2))
  return itensor(M, i1, i2)
end

convert_union_nothing(v::Vector{T}) where {T} = convert(Vector{Union{T,Nothing}}, v)

function interleave(xs...)
  nexts = convert_union_nothing(collect(Base.iterate.(xs)))
  res = Union{eltype.(xs)...}[]
  while any(!isnothing, nexts)
    for ii in eachindex(nexts)
      if !isnothing(nexts[ii])
        (item, state) = nexts[ii]
        push!(res, item)
        nexts[ii] = iterate(xs[ii], state)
      end
    end
  end
  return res
end

function correlation_matrix_to_mps(
  s::Vector{<:Index},
  Λ_up::AbstractMatrix,
  Λ_dn::AbstractMatrix;
  eigval_cutoff::Float64=1e-8,
  maxblocksize::Int=min(size(Λ_up, 1), size(Λ_dn, 1)),
  kwargs...,
)
  @assert size(Λ_up, 1) == size(Λ_up, 2)
  @assert size(Λ_dn, 1) == size(Λ_dn, 2)
  N_up = size(Λ_up, 1)
  N_dn = size(Λ_dn, 1)
  @assert N_up == N_dn
  # Total number of fermion sites
  N = N_up + N_dn
  ns_up, C_up = correlation_matrix_to_gmps(
    Λ_up; eigval_cutoff=eigval_cutoff, maxblocksize=maxblocksize
  )
  ns_dn, C_dn = correlation_matrix_to_gmps(
    Λ_dn; eigval_cutoff=eigval_cutoff, maxblocksize=maxblocksize
  )
  # map the up spins to the odd sites and the even spins to the even sites
  C_up = mapindex(n -> 2n - 1, C_up)
  C_dn = mapindex(n -> 2n, C_dn)
  C = Circuit(interleave(C_up.rotations, C_dn.rotations))
  ns = interleave(ns_up, ns_dn)
  if all(hastags("Fermion"), s)
    @assert length(s) == N
    U = [ITensor(s, g) for g in reverse(C.rotations)]
    ψ = MPS(s, n -> round(Int, ns[n]) + 1, U; kwargs...)
  elseif all(hastags("Electron"), s)
    @assert length(s) == N_up
    @assert length(s) == N_dn
    if isspinful(s)
      space_up = [QN(("Nf", 0, -1), ("Sz", 0)) => 1, QN(("Nf", 1, -1), ("Sz", 1)) => 1]
      space_dn = [QN(("Nf", 0, -1), ("Sz", 0)) => 1, QN(("Nf", 1, -1), ("Sz", -1)) => 1]
      sf_up = [Index(space_up, "Fermion,Site,n=$(2n-1)") for n in 1:N_up]
      sf_dn = [Index(space_dn, "Fermion,Site,n=$(2n)") for n in 1:N_dn]
      sf = collect(Iterators.flatten(zip(sf_up, sf_dn)))
    else
      sf = siteinds("Fermion", N; conserve_qns=true, conserve_sz=false)
    end
    U = [ITensor(sf, g) for g in reverse(C.rotations)]
    ψf = MPS(sf, n -> round(Int, ns[n]) + 1, U; kwargs...)
    ψ = MPS(N_up)
    for n in 1:N_up
      i, j = 2 * n - 1, 2 * n
      C = combiner(sf[i], sf[j])
      c = combinedind(C)
      ψ[n] = ψf[i] * ψf[j] * C
      ψ[n] *= identity_blocks_itensor(dag(c), s[n])
    end
  else
    error("All sites must be Fermion or Electron type.")
  end
  return ψ
end
