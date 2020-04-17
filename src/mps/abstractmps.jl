
abstract type AbstractMPS end

"""
    length(::MPS/MPO)

The number of sites of an MPS/MPO.
"""
Base.length(m::AbstractMPS) = m.length

"""
    ITensors.data(::MPS/MPO)

The Vector storage of an MPS/MPO.

This is mostly for internal usage, please let us
know if there is functionality not available for
MPS/MPO you would like.
"""
data(m::AbstractMPS) = m.data

leftlim(m::AbstractMPS) = m.llim

rightlim(m::AbstractMPS) = m.rlim

function setleftlim!(m::AbstractMPS, new_ll::Int)
  m.llim = new_ll
end

function setrightlim!(m::AbstractMPS, new_rl::Int)
  m.rlim = new_rl
end

isortho(m::AbstractMPS) = leftlim(m)+1 == rightlim(m)-1

function orthocenter(m::T) where {T<:AbstractMPS}
  !isortho(m) && error("$T has no well-defined orthogonality center")
  return leftlim(m)+1
end

Base.getindex(M::AbstractMPS,
              n::Integer) = getindex(data(M),n)

function Base.setindex!(M::AbstractMPS,
                        T::ITensor,
                        n::Integer)
  (n <= leftlim(M)) && setleftlim!(M,n-1)
  (n >= rightlim(M)) && setrightlim!(M,n+1)
  setindex!(data(M),T,n)
end

Base.copy(m::AbstractMPS) = typeof(m)(length(m),
                                      copy(data(m)),
                                      leftlim(m),
                                      rightlim(m))

Base.similar(m::AbstractMPS) = typeof(m)(length(m),
                                         similar(data(m)),
                                         0,
                                         length(m))

Base.deepcopy(m::AbstractMPS) = typeof(m)(length(m),
                                          deepcopy(data(m)),
                                          leftlim(m),
                                          rightlim(m))

Base.eachindex(m::AbstractMPS) = 1:length(m)

Base.iterate(M::AbstractMPS) = iterate(data(M))

Base.iterate(M::AbstractMPS, state) = iterate(data(M), state)

"""
    dag(m::MPS)

    dag(m::MPO)

Hermitian conjugation of a matrix product state or operator `m`.
"""
function dag(m::AbstractMPS)
  N = length(m)
  mdag = typeof(m)(N)
  for i in eachindex(m)
    mdag[i] = dag(m[i])
  end
  return mdag
end

"""
    prime!(m::MPS, args...; kwargs...)

    prime!(m::MPO, args...; kwargs...)

Prime all ITensors of an MPS/MPO in-place.
"""
function prime!(M::AbstractMPS, args...; kwargs...)
  for i in eachindex(M)
    M[i] = prime(M[i], args...; kwargs...)
  end
  return M
end

"""
    prime(m::MPS, args...; kwargs...)

    prime(m::MPO, args...; kwargs...)

Prime all ITensors of an MPS/MPO.
"""
prime(M::AbstractMPS, vargs...) = prime!(copy(M), vargs...)

"""
    noprime!(m::MPS, args...; kwargs...)

    noprime!(m::MPO, args...; kwargs...)

Set the prime level to zero for all ITensors of an MPS/MPO,
in-place.
"""
function noprime!(M::AbstractMPS, args...; kwargs...)
  for i in eachindex(M)
    M[i] = noprime(M[i], args...; kwargs...)
  end
  return M
end

"""
    noprime(m::MPS, args...; kwargs...)

    noprime(m::MPO, args...; kwargs...)

Set the prime level to zero for all ITensors of an MPS/MPO.
"""
noprime(M::AbstractMPS, vargs...) = noprime!(copy(M), vargs...)

function primelinkinds!(M::AbstractMPS, plinc::Integer = 1)
  for i ∈ eachindex(M)[1:end-1]
    l = linkind(M,i)
    prime!(M[i], plinc, l)
    prime!(M[i+1], plinc, l)
  end
end

function simlinkinds!(M::AbstractMPS)
  for i ∈ eachindex(M)[1:end-1]
    isnothing(commonind(M[i],M[i+1])) && continue
    l = linkind(M,i)
    l̃ = sim(l)
    replaceind!(M[i],l,l̃)
    replaceind!(M[i+1],l,dag(l̃))
  end
end

"""
    maxlinkdim(M::MPS)

    maxlinkdim(M::MPO)

Get the maximum link dimension of the MPS or MPO.
"""
function maxlinkdim(M::AbstractMPS)
  md = 0
  for b ∈ eachindex(M)[1:end-1]
    md = max(md,dim(linkind(M,b)))
  end
  md
end

function Base.show(io::IO, M::AbstractMPS)
  print(io,"$(typeof(M))")
  (length(M) > 0) && print(io,"\n")
  for (i, A) ∈ enumerate(data(M))
    if order(A) != 0
      println(io,"[$i] $(inds(A))")
    else
      println(io,"[$i] ITensor()")
    end
  end
end

function linkind(M::AbstractMPS, j::Int)
  N = length(M)
  j ≥ length(M) && error("No link index to the right of site $j (length of MPS is $N)")
  li = commonind(M[j],M[j+1])
  if isnothing(li)
    error("linkind: no MPS link index at link $j")
  end
  return li
end

function plussers(::Type{T},
                  left_ind::Index,
                  right_ind::Index,
                  sum_ind::Index) where {T<:Array}
  total_dim    = dim(left_ind) + dim(right_ind)
  total_dim    = max(total_dim, 1)
  # TODO: I am not sure if we should be using delta
  # tensors for this purpose? I think we should consider
  # not allowing them to be made with different index sizes
  #left_tensor  = δ(left_ind, sum_ind)
  left_tensor  = diagITensor(1.0,left_ind, sum_ind)
  right_tensor = ITensor(right_ind, sum_ind)
  for i in 1:dim(right_ind)
    right_tensor[right_ind(i), sum_ind(dim(left_ind) + i)] = 1
  end
  return left_tensor, right_tensor
end

"""
    add(A::MPS, B::MPS; kwargs...)
    +(A::MPS, B::MPS; kwargs...)

    add(A::MPO, B::MPO; kwargs...)
    +(A::MPO, B::MPO; kwargs...)

Add two MPS (or MPO) with each other, with some optional
truncation.
"""
function Base.:+(A::T, B::T; kwargs...) where {T <: AbstractMPS}
    N = length(A)
    length(B) != N && throw(DimensionMismatch("lengths of MPOs A ($N) and B ($(length(B))) do not match"))
    orthogonalize!(A, 1; kwargs...)
    orthogonalize!(B, 1; kwargs...)
    C = similar(A)
    rand_plev = 13124
    lAs = [linkind(A, i) for i in 1:N-1]
    prime!(A, rand_plev, "Link")

    first  = Vector{ITensor{2}}(undef,N-1)
    second = Vector{ITensor{2}}(undef,N-1)
    for i in 1:N-1
        lA = linkind(A, i)
        lB = linkind(B, i)
        r  = Index(dim(lA) + dim(lB), tags(lA))
        f, s = plussers(typeof(data(A[1])), lA, lB, r)
        first[i]  = f
        second[i] = s
    end
    C[1] = A[1] * first[1] + B[1] * second[1]
    for i in 2:N-1
        C[i] = dag(first[i-1]) * A[i] * first[i] + dag(second[i-1]) * B[i] * second[i]
    end
    C[N] = dag(first[N-1]) * A[N] + dag(second[N-1]) * B[N]
    prime!(C, -rand_plev, "Link")
    truncate!(C; kwargs...)
    return C
end

add(A::T, B::T;
    kwargs...) where {T <: AbstractMPS} = +(A, B; kwargs...)

"""
    sum(A::Vector{MPS}; kwargs...)

    sum(A::Vector{MPO}; kwargs...)

Add multiple MPS (or MPO) with each other, with some optional
truncation.
"""
function Base.sum(A::Vector{T};
                  kwargs...) where {T <: AbstractMPS}
  length(A) == 0 && return T()
  length(A) == 1 && return A[1]
  length(A) == 2 && return +(A[1], A[2]; kwargs...)
  nsize = isodd(length(A)) ? (div(length(A) - 1, 2) + 1) : div(length(A), 2)
  newterms = Vector{T}(undef, nsize)
  np = 1
  for n in 1:2:length(A) - 1
    newterms[np] = +(A[n], A[n+1]; kwargs...)
    np += 1
  end
  if isodd(length(A))
    newterms[nsize] = A[end]
  end
  return sum(newterms; kwargs...)
end

function orthogonalize!(M::AbstractMPS,
                        j::Int;
                        kwargs...)
  while leftlim(M) < (j-1)
    (leftlim(M) < 0) && setleftlim!(M, 0)
    b = leftlim(M)+1
    linds = uniqueinds(M[b],M[b+1])
    L,R = factorize(M[b], linds)
    M[b] = L
    M[b+1] *= R

    setleftlim!(M,b)
    if rightlim(M) < leftlim(M)+2
      setrightlim!(M, leftlim(M)+2)
    end
  end

  N = length(M)

  while rightlim(M) > (j+1)
    (rightlim(M) > (N+1)) && setrightlim!(M,N+1)
    b = rightlim(M)-2
    rinds = uniqueinds(M[b+1],M[b])
    L,R = factorize(M[b+1], rinds)
    M[b+1] = L
    M[b] *= R

    setrightlim!(M,b+1)
    if leftlim(M) > rightlim(M)-2
      setleftlim!(M, rightlim(M)-2)
    end
  end
end

function NDTensors.truncate!(M::AbstractMPS;
                             kwargs...)
  N = length(M)

  # Left-orthogonalize all tensors to make
  # truncations controlled
  orthogonalize!(M,N)

  # Perform truncations in a right-to-left sweep
  for j in reverse(2:N)
    rinds = uniqueinds(M[j],M[j-1])
    U,S,V = svd(M[j],rinds;kwargs...)
    M[j] = U
    M[j-1] *= (S*V)
    setrightlim!(M,j)
  end

end

mul(A::AbstractMPS, B::AbstractMPS; kwargs...) = *(A, B; kwargs...)

"""
    *(x::Number, M::MPS)

    *(x::Number, M::MPO)

Scale the MPS or MPO by the provided number.

Note: right now this just naively scales the
middle tensor. In the future, the plan would be
to scale the tensors between the left limit
and right limit by `x^(1/N)` where `N` is the distance 
between the left limit and right limit.
"""
function Base.:*(x::Number, M::AbstractMPS)
  N = deepcopy(M)
  c = div(length(N), 2)
  N[c] .*= x
  return N
end

Base.:-(M::AbstractMPS) = Base.:*(-1,M)

@doc """
    orthogonalize!(M::MPS, j::Int; kwargs...)

    orthogonalize!(M::MPO, j::Int; kwargs...)

Move the orthogonality center of the MPS
to site `j`. No observable property of the
MPS will be changed, and no truncation of the
bond indices is performed. Afterward, tensors
`1,2,...,j-1` will be left-orthogonal and tensors
`j+1,j+2,...,N` will be right-orthogonal.
""" orthogonalize!

@doc """
    truncate!(M::MPS; kwargs...)

    truncate!(M::MPO; kwargs...)

Perform a truncation of all bonds of an MPS/MPO,
using the truncation parameters (cutoff,maxdim, etc.)
provided as keyword arguments.
""" truncate!

@deprecate orthoCenter(args...; kwargs...) orthocenter(args...; kwargs...)

import .NDTensors.store
@deprecate store(m::AbstractMPS) data(m)

@deprecate replacesites!(args...; kwargs...) ITensors.replacesiteinds!(args...; kwargs...)

@deprecate applyMPO(args...; kwargs...) mul(args...; kwargs...)

@deprecate applympo(args...; kwargs...) mul(args...; kwargs...)

@deprecate errorMPOprod(args...; kwargs...) error_mul(args...; kwargs...)

@deprecate error_mpoprod(args...; kwargs...) error_mul(args...; kwargs...)

@deprecate multMPO(args...; kwargs...) mul(args...; kwargs...)

import Base.sum

@deprecate sum(A::AbstractMPS,
               B::AbstractMPS; kwargs...) add(A, B; kwargs...)

@deprecate multmpo(args...; kwargs...) mul(args...; kwargs...)

@deprecate set_leftlim!(args...; kwargs...) ITensors.setleftlim!(args...; kwargs...)

@deprecate set_rightlim!(args...; kwargs...) ITensors.setrightlim!(args...; kwargs...)

@deprecate tensors(args...; kwargs...) ITensors.data(args...; kwargs...)

@deprecate primelinks!(args...; kwargs...) ITensors.primelinkinds!(args...; kwargs...)

@deprecate simlinks!(args...; kwargs...) ITensors.simlinkinds!(args...; kwargs...)

