# brick wall scanning for a single MERA layer with treatment to the tail
function correlation_matrix_to_gmps_brickwall_tailed(
  Λ0::AbstractMatrix{ElT},
  inds::Vector{Int};
  eigval_cutoff::Float64=1e-8,
  maxblocksize::Int=size(Λ0, 1),
) where {ElT<:Number}
  Λ = Hermitian(Λ0)
  N = size(Λ, 1)
  V = Circuit{ElT}([])
  #ns = Vector{real(ElT)}(undef, 2*N)
  err_tot = 0.0
  indsnext = Int[]
  relinds = Int[]
  for i in 1:N
    if i % 2 == 0
      append!(indsnext, inds[i])
      append!(relinds, i)
      continue
    end
    blocksize = 0
    n = 0.0
    err = 0.0
    p = Int[]
    uB = 0.0
    # find the block whose lowest eigenvalue is within torelence
    for blocksize in 1:maxblocksize
      j = min(i + blocksize, N)
      ΛB = deepcopy(Λ[i:j, i:j]) #@view Λ[i:j, i:j] # \LambdaB is still part of Lambda
      nB, uB = eigen(Hermitian(ΛB))
      # sort by -(n * log(n) + (1 - n) * log(1 - n)) in ascending order
      p = sortperm(nB; by=entropy)
      n = nB[p[1]]
      err = min(n, 1 - n)
      err ≤ eigval_cutoff && break
    end
    # keep the node if the err cannot be reduced
    if i + maxblocksize >= N && err > eigval_cutoff
      append!(indsnext, inds[i])
      append!(relinds, i)
      continue
    end
    err_tot += err
    #ns[i] = n # eigenvalue
    v = deepcopy(uB[:, p[1]]) #@view uB[:, p[1]] # eigenvector of the correlation matrix
    g, _ = givens_rotations(v) # convert eigenvector into givens rotation
    shift!(g, i - 1) # shift rotation location
    # In-place version of:
    # V = g * V
    lmul!(g, V)
    #@show g
    Λ = Hermitian(g * Λ * g') #isolate current site i
  end
  return Λ, V, indsnext, relinds
end

# shift givens rotation indexes according to the inds 
function shiftByInds!(G::Circuit, inds::Vector{Int})
  for (n, g) in enumerate(G.rotations)
    G.rotations[n] = Givens(inds[g.i1], inds[g.i2], g.c, g.s)
  end
  return G
end

"""
    correlation_matrix_to_gmera(Λ::AbstractMatrix{ElT}; eigval_cutoff::Float64 = 1e-8, maxblocksize::Int = size(Λ0, 1))

Diagonalize a correlation matrix through MERA layers,
output gates and eigenvalues of the correlation matrix
"""
# Combine gates for each MERA layer
function correlation_matrix_to_gmera(
  Λ0::AbstractMatrix{ElT}; eigval_cutoff::Float64=1e-8, maxblocksize::Int=size(Λ0, 1)
) where {ElT<:Number}
  Λ = Hermitian(Λ0)
  N = size(Λ, 1)
  Nnew = N - 1
  inds = collect(1:N)
  V = Circuit{ElT}([])
  Λtemp = deepcopy(Λ)
  layer = 0                                 # layer label of MERA
  while N > Nnew # conditioned on the reduction of nodes
    N = Nnew
    # indsnext: next layer indexes with original matrix labels
    # relinds: next layer indexes with labels from the last layer
    Λr, C, indsnext, relinds = correlation_matrix_to_gmps_brickwall_tailed(
      Λtemp, inds; eigval_cutoff=eigval_cutoff, maxblocksize=maxblocksize
    )
    shiftByInds!(C, inds)    # shift the index back to the original matrix
    inds = indsnext
    Λtemp = deepcopy(Λr[relinds, relinds]) # project to even site for next layer based on keeping indexes relinds
    Nnew = size(Λtemp, 1)
    lmul!(C, V)                           # add vector of givens rotation C into the larger vector V
    #V = C * V
    layer += 1
    #Λ = ITensors.Hermitian(C * Λ * C')
  end
  # gmps for the final layer
  Λr, C = correlation_matrix_to_gmps(
    Λtemp; eigval_cutoff=eigval_cutoff, maxblocksize=maxblocksize
  )
  shiftByInds!(C, inds)
  lmul!(C, V)
  Λ = V * Λ0 * V'
  ns = real.(diag(Λ))
  return ns, V
end

# output the MERA gates and eigenvalues of correlation matrix from WF
function slater_determinant_to_gmera(Φ::AbstractMatrix; kwargs...)
  return correlation_matrix_to_gmera(conj(Φ) * transpose(Φ); kwargs...)
end

# ouput the MPS based on the MERA gates
function correlation_matrix_to_mera(
  s::Vector{<:Index},
  Λ::AbstractMatrix;
  eigval_cutoff::Float64=1e-8,
  maxblocksize::Int=size(Λ, 1),
  kwargs...,
)
  @assert size(Λ, 1) == size(Λ, 2)
  ns, C = correlation_matrix_to_gmera(
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

function slater_determinant_to_mera(s::Vector{<:Index}, Φ::AbstractMatrix; kwargs...)
  return correlation_matrix_to_mera(s, conj(Φ) * transpose(Φ); kwargs...)
end

# G the circuit from the gates, N is the total number of sites
function UmatFromGates(G::Circuit, N::Int)
  U = Matrix{Float64}(I, N, N)
  n = size(G.rotations, 1)
  for k in 1:n
    rot = G.rotations[k]
    U = rot * U
  end
  return U
end

# compute the energy of the state based on the gates
function EfromGates(H::Matrix{<:Number}, U::Matrix{<:Number})
  Htemp = U * H * U'
  Etot = 0
  N = size(U, 1)
  for i in 1:N
    if Htemp[i, i] < 0.0
      Etot += Htemp[i, i]
    end
  end
  return Etot
end
