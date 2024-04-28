
#
# `depth_first_constructive` is a very simple recursive implementation
# but it is more difficult to cap the costs so scales very badly
#

function depth_first_constructive(T::Vector{<:ITensor}) where {LabelSetT}
  indsT = [inds(Tₙ) for Tₙ in T]
  return depth_first_constructive(DimT, indsT)
end

function depth_first_constructive(
  ::Type{DimT}, T::Vector{IndexSetT}
) where {IndexSetT<:IndexSet,DimT}
  labels, dims = label_dims(DimT, T)
  nlabels = length(dims)
  if nlabels ≤ 16
    return depth_first_constructive(UInt16, labels, dims)
  elseif nlabels ≤ 32
    return depth_first_constructive(UInt32, labels, dims)
  elseif nlabels ≤ 64
    return depth_first_constructive(UInt64, labels, dims)
  elseif nlabels ≤ 128
    return depth_first_constructive(UInt128, labels, dims)
  else
    return depth_first_constructive(BitSet, labels, dims)
  end
end

function depth_first_constructive(
  ::Type{LabelSetT}, labels::Vector, dims::Vector
) where {LabelSetT}
  return depth_first_constructive(map(label -> bitset(LabelSetT, label), labels), dims)
end

function depth_first_constructive(
  ::Type{LabelSetT}, ::Type{DimT}, T::Vector{<:ITensor}
) where {LabelSetT,DimT}
  indsT = [inds(Tₙ) for Tₙ in T]
  return depth_first_constructive(LabelSetT, DimT, indsT)
end

function depth_first_constructive(
  ::Type{LabelSetT}, ::Type{DimT}, T::Vector{IndexSetT}
) where {IndexSetT<:IndexSet,LabelSetT,DimT}
  labels, dims = label_dims(DimT, T)
  return depth_first_constructive(map(label -> bitset(LabelSetT, label), labels), dims)
end

function depth_first_constructive(T::Vector, ind_dims::Vector)
  optimal_cost = Ref(typemax(eltype(ind_dims)))
  optimal_sequence = Vector{Pair{Int,Int}}(undef, length(T) - 1)
  _depth_first_constructive!(
    optimal_sequence, optimal_cost, Pair{Int,Int}[], T, ind_dims, collect(1:length(T)), 0
  )
  return pair_sequence_to_tree(optimal_sequence, length(T))
end

function _depth_first_constructive!(
  optimal_sequence, optimal_cost, sequence, T, ind_dims, remaining, cost
)
  if length(remaining) == 1
    # Only should get here if the contraction was the best
    # Otherwise it would have hit the `continue` below
    @assert cost ≤ optimal_cost[]
    optimal_cost[] = cost
    optimal_sequence .= sequence
  end
  for aᵢ in 1:(length(remaining) - 1), bᵢ in (aᵢ + 1):length(remaining)
    a = remaining[aᵢ]
    b = remaining[bᵢ]
    current_cost, Tᵈ = contraction_cost(T[a], T[b], ind_dims)
    new_cost = cost + current_cost
    if new_cost ≥ optimal_cost[]
      continue
    end
    new_sequence = push!(copy(sequence), a => b)
    new_T = push!(copy(T), Tᵈ)
    new_remaining = deleteat!(copy(remaining), (aᵢ, bᵢ))
    push!(new_remaining, length(new_T))
    _depth_first_constructive!(
      optimal_sequence, optimal_cost, new_sequence, new_T, ind_dims, new_remaining, new_cost
    )
  end
end
