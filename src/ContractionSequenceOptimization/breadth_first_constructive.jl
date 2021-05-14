
#
# Breadth-first constructive approach
#

function breadth_first_constructive(indsT::Vector)
  ntensors = length(indsT)
  if ntensors ≤ 16
    return breadth_first_constructive(UInt16, DimT, indsT)
  elseif ntensors ≤ 32
    return breadth_first_constructive(UInt32, DimT, indsT)
  elseif ntensors ≤ 64
    return breadth_first_constructive(UInt64, DimT, indsT)
  elseif ntensors ≤ 128
    return breadth_first_constructive(UInt128, DimT, indsT)
  else
    return breadth_first_constructive(BitSet, DimT, indsT)
  end
end

breadth_first_constructive(T::Tuple) = breadth_first_constructive(collect(T))

function breadth_first_constructive(
  ::Type{TensorSetT}, ::Type{DimT}, T::Vector{IndexSetT}
) where {IndexSetT,TensorSetT,DimT}
  labels, alldims = label_dims(DimT, T)
  nlabels = length(alldims)
  if nlabels ≤ 16
    return breadth_first_constructive(TensorSetT, UInt16, labels, alldims)
  elseif nlabels ≤ 32
    return breadth_first_constructive(TensorSetT, UInt32, labels, alldims)
  elseif nlabels ≤ 64
    return breadth_first_constructive(TensorSetT, UInt64, labels, alldims)
  elseif nlabels ≤ 128
    return breadth_first_constructive(TensorSetT, UInt128, labels, alldims)
  else
    return breadth_first_constructive(TensorSetT, BitSet, labels, alldims)
  end
end

function breadth_first_constructive(
  ::Type{TensorSetT}, ::Type{LabelSetT}, labels::Vector, alldims::Vector
) where {TensorSetT,LabelSetT}
  return breadth_first_constructive(
    TensorSetT, map(label -> bitset(LabelSetT, label), labels), alldims
  )
end

# TODO: delete?
#function breadth_first_constructive(::Type{TensorSetT}, ::Type{LabelSetT}, ::Type{DimT},
#                                    T::Vector{<: ITensor}) where {TensorSetT, LabelSetT, DimT}
#  indsT = [inds(Tₙ) for Tₙ in T]
#  return breadth_first_constructive(TensorSetT, LabelSetT, DimT, indsT)
#end

function breadth_first_constructive(
  ::Type{TensorSetT}, ::Type{LabelSetT}, ::Type{DimT}, T::Vector{IndexSetT}
) where {IndexSetT,TensorSetT,LabelSetT,DimT}
  labels, alldims = label_dims(DimT, T)
  return breadth_first_constructive(
    TensorSetT, map(label -> bitset(LabelSetT, label), labels), alldims
  )
end

# A type storing information about subnetworks
const SubNetwork{LabelSetT,DimT} = NamedTuple{
  (:inds, :cost, :sequence),Tuple{LabelSetT,DimT,Vector{Any}}
}

function breadth_first_constructive(
  ::Type{TensorSetT}, T::Vector{LabelSetT}, alldims::Vector{DimT}
) where {TensorSetT,LabelSetT,DimT}
  components = connectedcomponents(T, alldims)
  N = length(components)
  if N == 1
    return _breadth_first_constructive(TensorSetT, collect(1:length(T)), T, alldims)
  end
  sequences = Vector{Any}(undef, N)
  for n in 1:N
    componentsₙ = components[n]
    if length(componentsₙ) == 1
      sequences[n] = only(componentsₙ)
      continue
    elseif length(componentsₙ) == 2
      sequences[n] = componentsₙ
      continue
    end
    sequences[n] = _breadth_first_constructive(
      TensorSetT, componentsₙ, T[componentsₙ], alldims
    )
  end
  return sequences
end

# Apply breadth_first_constructive to a single disconnected subnetwork
# Based on: https://arxiv.org/abs/1304.6112 and https://github.com/Jutho/TensorOperations.jl/blob/v3.1.0/src/indexnotation/optimaltree.jl
function _breadth_first_constructive(
  ::Type{TensorSetT}, Tlabels::Vector, T::Vector{LabelSetT}, alldims::Vector{DimT}
) where {TensorSetT,LabelSetT,DimT}
  n = length(T)

  # `cache[c]` is the set of all objects made up by
  # contracting `c` unique tensors from the original tensors `1:n`.
  cache = Vector{Dict{TensorSetT,SubNetwork{LabelSetT,DimT}}}(undef, n)
  for c in 1:n
    # Initialized to empty
    cache[c] = eltype(cache)()
  end
  # Fill the first cache with trivial data
  for i in 1:n
    cache[1][bitset(TensorSetT, [Tlabels[i]])] = (inds=T[i], cost=0, sequence=Any[])
  end

  # TODO: pick a reasonable maxcost, the product of all dimensions
  # of tensors in the network. Could also be `typemax(DimT)`?
  maxcost = try
    # This may overflow, so we catch the error and return typemax(DimT)
    Base.Checked.checked_mul(
      reduce(Base.Checked.checked_mul, alldims; init=one(DimT)), maximum(alldims)
    )
  catch
    typemax(DimT)
  end

  # TODO: pick a reasonable initialcost
  # Maybe use the cost of the trivial contraction [4, [3, [2, 1]]]?
  tensordims = Vector{DimT}(undef, n)
  for k in 1:n
    tensordims[k] = dim(T[k], alldims)
  end
  _initialcost, overflow = Base.Checked.mul_with_overflow(
    maximum(tensordims), minimum(tensordims)
  )
  _initialcost = overflow ? typemax(DimT) : _initialcost
  initialcost = min(maxcost, _initialcost)

  # Factor to increase the cost cap by each iteration
  costfac = maximum(alldims)

  currentcost = initialcost
  previouscost = zero(initialcost)

  while isempty(cache[n])
    nextcost = maxcost

    # c is the total number of tensors being contracted
    # in the current sequence
    for c in 2:n
      # For each pair of sets Sᵈ, Sᶜ⁻ᵈ, 1 ≤ d ≤ ⌊c/2⌋
      for d in 1:(c ÷ 2)
        for a in keys(cache[d]), b in keys(cache[c - d])
          if d == c - d && _isless(b, a)
            # When d == c-d (the subset sizes are equal), check that
            # b > a so that that case (a,b) and (b,a) are not repeated
            continue
          end

          if !_isemptyset(_intersect(a, b))
            # Check that each element of S¹ appears
            # at most once in (TᵃTᵇ).
            continue
          end

          # Use previously computed cost of contracting network `ab` and compare against the previouscost
          ab = _union(a, b)
          cache_c = @inbounds cache[c]
          cache_ab = get(cache_c, ab, nothing)
          currentcost_ab = isnothing(cache_ab) ? currentcost : cache_ab.cost
          if currentcost_ab ≤ previouscost
            continue
          end

          # Determine the cost μ of contracting Tᵃ, Tᵇ
          # These dictionary calls and `contraction_cost` take
          # up most of the time.
          cache_a = cache[d][a]
          cache_b = cache[c - d][b]

          if dim(_intersect(cache_a.inds, cache_b.inds), alldims) < 2
            # XXX: For now, ignore outer products contractions.
            # In the future, handle this in a more sophisticated way.
            continue
          end

          cost, inds_ab = contraction_cost(cache_a.inds, cache_b.inds, alldims)
          if iszero(cost)
            # If the cost is zero, that means the multiplication overflowed
            continue
          end

          if d > 1
            # Add to cost of contracting the subnetwork `a`
            cost, overflow = Base.Checked.add_with_overflow(cost, cache_a.cost)
            overflow && continue
          end
          if c - d > 1
            # Add to cost of contracting the subnetwork `b`
            cost, overflow = Base.Checked.add_with_overflow(cost, cache_b.cost)
            overflow && continue
          end

          if cost ≤ currentcost_ab
            cost_ab = cost
            if d == 1
              sequence_a = _only(a)
            else
              sequence_a = cache_a.sequence
            end
            if c - d == 1
              sequence_b = _only(b)
            else
              sequence_b = cache_b.sequence
            end
            sequence_ab = Any[sequence_a, sequence_b]

            # XXX: this call is pretty slow (maybe takes 1/3 of total time in large n limit)
            cache_c[ab] = (inds=inds_ab, cost=cost_ab, sequence=sequence_ab)
          end # if cost ≤ currentcost_ab
        end # for a in S[d], b in S[c-d]
      end # for d in 1:c÷2
    end # for c in 2:n
    previouscost = currentcost
    currentcost = min(maxcost, nextcost * costfac)

    # Reset all tensors to old
    for i in 1:n
      for a in eachindex(cache[i])
        cache_a = cache[i][a]
        cache[i][a] = (inds=cache_a.inds, cost=cache_a.cost, sequence=cache_a.sequence)
      end
    end
  end # while isempty(S[n])
  Sⁿ = bitset(TensorSetT, Tlabels)
  return cache[n][Sⁿ].sequence
end
