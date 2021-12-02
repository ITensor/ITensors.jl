function cuMPS(psi::MPS)
    phi = copy(psi)
    for site in 1:length(psi)
        phi.data[site] = cuITensor(psi.data[site])
    end
    return phi
end

cu(ψ::MPS) = cuMPS(ψ)

cuMPS() = MPS() 
function cuMPS(::Type{T}, sites::Vector{<:Index}; linkdims::Integer=1) where {T<:Number}
  N = length(sites)
  v = Vector{ITensor}(undef, N)
  if N == 1
    v[1] = emptyITensor(T, sites[1])
    return MPS(v)
  end

  space = if hasqns(sites)
    [QN() => linkdims]
  else
    linkdims
  end
  
  l = [Index(1, "Link,l=$ii") for ii=1:N-1]
  for ii in eachindex(sites)
    s = sites[ii]
    if ii == 1
      v[ii] = cuITensor(l[ii], s)
    elseif ii == N
      v[ii] = cuITensor(l[ii-1], s)
    else
      v[ii] = cuITensor(l[ii-1],s,l[ii])
    end
  end
  return MPS(v,0,N+1)
end
cuMPS(sites::Vector{<:Index}, args...; kwargs...) = cuMPS(Float64, sites, args...; kwargs...)
function cuMPS(N::Int; ortho_lims::UnitRange=1:N)
    return MPS(Vector{ITensor}(undef, N); ortho_lims=ortho_lims)
end

function randomCuMPS(sites)
  M = cuMPS(sites)
  for i in eachindex(sites)
    randn!(M[i])
    normalize!(M[i])
  end
  M.llim = 1
  M.rlim = length(M)
  return M
end
function randomCuMPS(N::Int; ortho_lims::UnitRange=1:N)
  return randomCuMPS(Vector{ITensor}(undef, N); ortho_lims=ortho_lims)
end

const productCuMPS = cuMPS

function cuMPS(::Type{T}, ivals::Vector{<:Pair{<:Index}}) where {T<:Number}
  N     = length(ivals)
  As    = Vector{ITensor}(undef,N)
  links = Vector{Index}(undef,N)
  for n=1:N
    s = ITensors.ind(ivals[n])
    links[n] = Index(1,"Link,l=$n")
    if n == 1
      A = ITensor(T, s, links[n])
      A[ivals[n],links[n](1)] = 1.0
    elseif n == N
      A = ITensor(T, links[n-1], s)
      A[links[n-1](1),ivals[n]] = 1.0
    else
      A = ITensor(T, links[n-1], s, links[n])
      A[links[n-1](1),ivals[n],links[n](1)] = 1.0
    end
    As[n] = cuITensor(A)
  end
  return MPS(As,0,2)
end
cuMPS(ivals::Vector{Pair{<:Index}}) = cuMPS(Float64, ivals)

function cuMPS(::Type{T}, sites::Vector{<:Index}, states_) where {T<:Number}
  if length(sites) != length(states_)
    throw(DimensionMismatch("Number of sites and and initial vals don't match"))
  end
  N = length(states_)
  M = cuMPS(N)

  if N == 1
    M[1] = state(sites[1], states_[1])
    return M
  end

  states = [state(sites[j], states_[j]) for j in 1:N]

  if hasqns(states[1])
    lflux = QN()
    for j in 1:(N - 1)
      lflux += flux(states[j])
    end
    links = Vector{QNIndex}(undef, N - 1)
    for j in (N - 1):-1:1
      links[j] = Index(lflux => 1; tags="Link,l=$j", dir=In)
      lflux -= flux(states[j])
    end
  else
    links = [Index(1; tags="Link,l=$n") for n in 1:N]
  end

  M[1] = cuITensor(T, sites[1], links[1])
  M[1] += cuITensor(states[1] * state(links[1], 1))
  for n in 2:(N - 1)
    M[n] = cuITensor(T, dag(links[n - 1]), sites[n], links[n])
    M[n] += cuITensor(state(dag(links[n - 1]), 1) * states[n] * state(links[n], 1))
  end
  M[N] = cuITensor(T, dag(links[N - 1]), sites[N])
  M[N] += cuITensor(state(dag(links[N - 1]), 1) * states[N])

  return M
end

function cuMPS(
  ::Type{T}, sites::Vector{<:Index}, state::Union{String,Integer}
) where {T<:Number}
  return cuMPS(T, sites, fill(state, length(sites)))
end

function cuMPS(::Type{T}, sites::Vector{<:Index}, states::Function) where {T<:Number}
  states_vec = [states(n) for n in 1:length(sites)]
  return cuMPS(T, sites, states_vec)
end

cuMPS(sites::Vector{<:Index}, states) = cuMPS(Float64, sites, states)
