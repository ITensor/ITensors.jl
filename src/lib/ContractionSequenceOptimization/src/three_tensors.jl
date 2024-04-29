
#
# Special case for three tensors
#

function compute_cost(external_dims::Tuple{Int,Int,Int}, internal_dims::Tuple{Int,Int,Int})
  dim11, dim22, dim33 = external_dims
  dim12, dim23, dim31 = internal_dims
  cost12 = dim11 * dim22 * dim12 * dim23 * dim31
  return cost12 + dim11 * dim22 * dim33 * dim31 * dim23
end

function three_tensor_contraction_sequence(which_sequence::Int)::Vector{Any}
  @assert 1 ≤ which_sequence ≤ 3
  return if which_sequence == 1
    Any[3, [1, 2]]
  elseif which_sequence == 2
    Any[1, [2, 3]]
  else
    Any[2, [3, 1]]
  end
end

function optimal_contraction_sequence(is1, is2, is3)
  N1 = length(is1)
  N2 = length(is2)
  N3 = length(is3)
  dim2 = dim(is2)
  dim3 = dim(is3)
  dim11 = 1
  dim12 = 1
  dim31 = 1
  @inbounds for n1 in 1:N1
    i1 = is1[n1]
    n2 = findfirst(==(i1), is2)
    if isnothing(n2)
      n3 = findfirst(==(i1), is3)
      if isnothing(n3)
        dim11 *= dim(i1)
        continue
      end
      dim31 *= dim(i1)
      continue
    end
    dim12 *= dim(i1)
  end
  dim23 = 1
  @inbounds for n2 in 1:length(is2)
    i2 = is2[n2]
    n3 = findfirst(==(i2), is3)
    if !isnothing(n3)
      dim23 *= dim(i2)
    end
  end
  dim22 = dim2 ÷ (dim12 * dim23)
  dim33 = dim3 ÷ (dim23 * dim31)
  external_dims1 = (dim11, dim22, dim33)
  internal_dims1 = (dim12, dim23, dim31)
  external_dims2 = (dim22, dim33, dim11)
  internal_dims2 = (dim23, dim31, dim12)
  external_dims3 = (dim33, dim11, dim22)
  internal_dims3 = (dim31, dim12, dim23)
  cost1 = compute_cost(external_dims1, internal_dims1)
  cost2 = compute_cost(external_dims2, internal_dims2)
  cost3 = compute_cost(external_dims3, internal_dims3)
  mincost, which_sequence = findmin((cost1, cost2, cost3))
  sequence = three_tensor_contraction_sequence(which_sequence)
  return sequence
end
