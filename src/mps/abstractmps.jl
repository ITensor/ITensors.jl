
abstract type AbstractMPS end

"""
    length(::MPS/MPO)

The number of sites of an MPS/MPO.
"""
Base.length(m::AbstractMPS) = length(m.data)

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
                        n::Integer;
                        set_limits::Bool = true)
  if set_limits
    (n <= leftlim(M)) && setleftlim!(M,n-1)
    (n >= rightlim(M)) && setrightlim!(M,n+1)
  end
  data(M)[n] = T
  return M
end

Base.copy(m::AbstractMPS) = typeof(m)(copy(data(m)),
                                      leftlim(m),
                                      rightlim(m))

Base.similar(m::AbstractMPS) = typeof(m)(similar(data(m)),
                                         0,
                                         length(m))

Base.deepcopy(m::AbstractMPS) = typeof(m)(deepcopy(data(m)),
                                          leftlim(m),
                                          rightlim(m))

Base.eachindex(m::AbstractMPS) = 1:length(m)

Base.iterate(M::AbstractMPS) = iterate(data(M))

Base.iterate(M::AbstractMPS, state) = iterate(data(M), state)

"""
    unique_siteind(A::MPO, B::MPS, j::Int)
    unique_siteind(A::MPO, B::MPO, j::Int)

Get the site index of MPO `A` that is unique to `A` (not shared with MPS/MPO `B`).
"""
function unique_siteind(A::AbstractMPS, B::AbstractMPS, j::Int)
  N = length(A)
  j == 1 && return uniqueind(A[j], A[j+1], B[j])
  j == N && return uniqueind(A[j], A[j-1], B[j])
  return uniqueind(A[j], A[j-1], A[j+1], B[j])
end

"""
    unique_siteinds(A::MPO, B::MPS)
    unique_siteinds(A::MPO, B::MPO)

Get the site indices of MPO `A` that are unique to `A` (not shared with MPS/MPO `B`), as a `Vector{<:Index}`.
"""
function unique_siteinds(A::AbstractMPS, B::AbstractMPS)
  return [unique_siteind(A, B, j) for j in eachindex(A)]
end

"""
    common_siteind(A::MPO, B::MPS, j::Int)
    common_siteind(A::MPO, B::MPO, j::Int)

Get the site index of MPO `A` that is shared with MPS/MPO `B`.
"""
function common_siteind(A::AbstractMPS, B::AbstractMPS, j::Int)
  return commonind(A[j], B[j])
end

"""
    common_siteinds(A::MPO, B::MPS)
    common_siteinds(A::MPO, B::MPO)

Get the site indices of MPO `A` that are shared with MPS/MPO `B`, as a `Vector{<:Index}`.
"""
function common_siteinds(A::AbstractMPS, B::AbstractMPS)
  return [common_siteind(A, B, j) for j in eachindex(A)]
end

function Base.map!(f::Function, M::AbstractMPS)
  for i in eachindex(M)
    setindex!(M, f(M[i]), i; set_limits = false)
  end
  return M
end

Base.map(f::Function, M::AbstractMPS) = map!(f, copy(M))

for fname in (:dag,
              :prime,
              :setprime,
              :noprime,
              :addtags,
              :removetags,
              :replacetags,
              :settags)
  @eval begin
    """
        $($fname)(M::MPS, args...; kwargs...)

        $($fname)(M::MPO, args...; kwargs...)

    Apply $($fname) to all ITensors of an MPS/MPO, returning a new MPS/MPO.

    The ITensors of the MPS/MPO will be a view of the storage of the original ITensors.
    """
    $fname(M::AbstractMPS,
           args...;
           kwargs...) = map(m -> $fname(m, args...;
                                        kwargs...), M)

    """
        $($fname)!(M::MPS, args...; kwargs...)

        $($fname)!(M::MPO, args...; kwargs...)

    Apply $($fname) to all ITensors of an MPS/MPO in-place.
    """
    $(Symbol(fname, :!))(M::AbstractMPS,
                         args...;
                         kwargs...) = map!(m -> $fname(m, args...;
                                                       kwargs...), M)

  end
end

function map_linkinds!(f::Function, M::AbstractMPS)
  for i in eachindex(M)[1:end-1]
    l = linkind(M, i)
    if !isnothing(l)
      l̃ = f(l)
      setindex!(M, replaceind(M[i], l, l̃), i;
                set_limits = false)
      setindex!(M, replaceind(M[i+1], l, l̃), i+1;
                set_limits = false)
    end
  end
  return M
end

map_linkinds(f::Function, M::AbstractMPS) = map_linkinds!(f, copy(M))

function map_common_siteinds!(f::Function, M1::AbstractMPS,
                                           M2::AbstractMPS)
  length(M1) != length(M2) && error("MPOs/MPSs must be the same length")
  for i in eachindex(M1)
    s = common_siteind(M1, M2, i)
    if !isnothing(s)
      s̃ = f(s)
      setindex!(M1, replaceind(M1[i], s, s̃), i;
                set_limits = false)
      setindex!(M2, replaceind(M2[i], s, s̃), i;
                set_limits = false)
    end
  end
  return M1, M2
end

function map_common_siteinds(f::Function, M1::AbstractMPS,
                                          M2::AbstractMPS)
  return map_common_siteinds!(f, copy(M1), copy(M2))
end

function map_unique_siteinds!(f::Function, M1::AbstractMPS,
                                           M2::AbstractMPS)
  length(M1) != length(M2) && error("MPOs/MPSs must be the same length")
  for i in eachindex(M1)
    s = unique_siteind(M1, M2, i)
    if !isnothing(s)
      s̃ = f(s)
      setindex!(M1, replaceind(M1[i], s, s̃), i;
                set_limits = false)
    end
  end
  return M1
end

function map_unique_siteinds(f::Function, M1::AbstractMPS,
                                        M2::AbstractMPS)
  return map_unique_siteinds!(f, copy(M1), M2)
end

for fname in (:sim,
              :prime,
              :setprime,
              :noprime,
              :addtags,
              :removetags,
              :replacetags,
              :settags)
  fname_linkinds = Symbol(fname, :_linkinds)
  fname_linkinds_inplace = Symbol(fname_linkinds, :!)
  fname_common_siteinds = Symbol(fname, :_common_siteinds)
  fname_common_siteinds_inplace = Symbol(fname_common_siteinds, :!)
  fname_unique_siteinds = Symbol(fname, :_unique_siteinds)
  fname_unique_siteinds_inplace = Symbol(fname_unique_siteinds, :!)

  @eval begin
    """
        $($fname_linkinds)(M::MPS, args...; kwargs...)

        $($fname_linkinds)(M::MPO, args...; kwargs...)

    Apply $($fname) to all link indices of an MPS/MPO, returning a new MPS/MPO.
    
    The ITensors of the MPS/MPO will be a view of the storage of the original ITensors.
    """
    $fname_linkinds(M::AbstractMPS,
                    args...;
                    kwargs...) = map_linkinds(i -> $fname(i, args...;
                                                         kwargs...), M)

    """
        $($fname_linkinds)!(M::MPS, args...; kwargs...)

        $($fname_linkinds)!(M::MPO, args...; kwargs...)

    Apply $($fname) to all link indices of the ITensors of an MPS/MPO in-place.
    """
    function $fname_linkinds_inplace(M::AbstractMPS,
                                     args...;
                                     kwargs...)
      return map_linkinds!(i -> $fname(i, args...;
                                      kwargs...), M)
    end

    """
        $($fname_common_siteinds)(M1::MPO, M2::MPS, args...; kwargs...)

        $($fname_common_siteinds)(M1::MPO, M2::MPO, args...; kwargs...)

    Apply $($fname) to the site indices that are shared by `M1` and `M2`.
    
    Returns new MPSs/MPOs. The ITensors of the MPSs/MPOs will be a view of the storage of the original ITensors.
    """
    function $fname_common_siteinds(M1::AbstractMPS,
                                    M2::AbstractMPS,
                                    args...;
                                    kwargs...)
      return map_common_siteinds(i -> $fname(i, args...;
                                             kwargs...), M1, M2)
    end

    """
        $($fname_common_siteinds)!(M1::MPO, M2::MPS, args...; kwargs...)

        $($fname_common_siteinds)!(M1::MPO, M2::MPO, args...; kwargs...)

    Apply $($fname) to the site indices that are shared by `M1` and `M2`. Returns new MPSs/MPOs.
    
    Modifies the input MPSs/MPOs in-place.
    """
    function $fname_common_siteinds_inplace(M1::AbstractMPS,
                                            M2::AbstractMPS,
                                            args...;
                                            kwargs...)
      return map_common_siteinds!(i -> $fname(i, args...;
                                              kwargs...), M1, M2)
    end

    """
        $($fname_unique_siteinds)(M1::MPO, M2::MPS, args...; kwargs...)

    Apply $($fname) to the site indices of `M1` that are not shared with `M2`. Returns new MPSs/MPOs.
    
    The ITensors of the MPSs/MPOs will be a view of the storage of the original ITensors.
    """
    function $fname_unique_siteinds(M1::AbstractMPS,
                                    M2::AbstractMPS,
                                    args...;
                                    kwargs...)
      return map_unique_siteinds(i -> $fname(i, args...;
                                             kwargs...), M1, M2)
    end

    """
        $($fname_unique_siteinds)!(M1::MPO, M2::MPS, args...; kwargs...)

    Apply $($fname) to the site indices of `M1` that are not shared with `M2`. Modifies the input MPSs/MPOs in-place.
    """
    function $fname_unique_siteinds_inplace(M1::AbstractMPS,
                                            M2::AbstractMPS,
                                            args...;
                                            kwargs...)
      return map_unique_siteinds!(i -> $fname(i, args...;
                                              kwargs...), M1, M2)
    end
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
    l = linkind(M, b)
    linkdim = isnothing(l) ? 0 : dim(l)
    md = max(md, linkdim)
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
  j ≥ length(M) && return nothing
  return commonind(M[j], M[j+1])
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

Add two MPS/MPO with each other, with some optional
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

Add multiple MPS/MPO with each other, with some optional
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

NDTensors.contract(A::AbstractMPS,
                   B::AbstractMPS;
                   kwargs...) = *(A, B; kwargs...)

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

"""
    hasqns(M::MPS)

    hasqns(M::MPO)

Return true if the MPS or MPO has
tensors which carry quantum numbers.
"""
hasqns(M::AbstractMPS) = hasqns(M[1])

"""
    flux(M::MPS)

    flux(M::MPO)

    totalqn(M::MPS)

    totalqn(M::MPO)

For an MPS or MPO which conserves quantum
numbers, compute the total QN flux. For
a tensor network such as an MPS or MPO,
the flux is the sum of fluxes of each of
the tensors in the network. The name
`totalqn` is an alias for `flux`.
"""
function flux(M::AbstractMPS)::QN
  hasqns(M) || error("MPS or MPO does not conserve QNs")
  q = QN()
  for j=M.llim+1:M.rlim-1
    q += flux(M[j])
  end
  return q
end

totalqn(M::AbstractMPS) = flux(M)

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

@deprecate orthoCenter(args...;
                       kwargs...) orthocenter(args...; kwargs...)

import .NDTensors.store

@deprecate store(m::AbstractMPS) data(m)

@deprecate replacesites!(args...;
                         kwargs...) ITensors.replace_siteinds!(args...; kwargs...)

@deprecate applyMPO(args...; kwargs...) contract(args...; kwargs...)

@deprecate applympo(args...; kwargs...) contract(args...; kwargs...)

@deprecate errorMPOprod(args...;
                        kwargs...) error_contract(args...;
                                                  kwargs...)

@deprecate error_mpoprod(args...;
                         kwargs...) error_contract(args...;
                                                   kwargs...)

@deprecate error_mul(args...;
                     kwargs...) error_contract(args...;
                                               kwargs...)

@deprecate multMPO(args...; kwargs...) contract(args...; kwargs...)

import Base.sum

@deprecate sum(A::AbstractMPS,
               B::AbstractMPS; kwargs...) add(A, B; kwargs...)

@deprecate multmpo(args...;
                   kwargs...) contract(args...; kwargs...)

@deprecate set_leftlim!(args...;
                        kwargs...) ITensors.setleftlim!(args...;
                                                        kwargs...)

@deprecate set_rightlim!(args...;
                         kwargs...) ITensors.setrightlim!(args...;
                                                          kwargs...)

@deprecate tensors(args...;
                   kwargs...) ITensors.data(args...; kwargs...)

@deprecate primelinks!(args...;
                       kwargs...) ITensors.prime_linkinds!(args...;
                                                          kwargs...)

@deprecate simlinks!(args...;
                     kwargs...) ITensors.sim_linkinds!(args...;
                                                      kwargs...)

@deprecate mul(A::AbstractMPS,
               B::AbstractMPS;
               kwargs...) contract(A, B; kwargs...)

