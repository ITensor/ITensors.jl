
left_associative_contraction_sequence(N::Integer) = reduce((x, y) -> Any[y, x], 1:N)
left_associative_contraction_sequence(A) = left_associative_contraction_sequence(length(A))

"""
    contraction_cost(A; sequence)

Return the cost of contracting the collection of ITensors according to the specified sequence,
where the cost is measured in the number of floating point operations that would need to
be performed to contract dense tensors of the dimensions specified by the indices of the tensors
(so for now, sparsity is ignored in computing the costs).
Pairwise costs are returned in a vector (contracting `N` tensors requires `N-1` pairwise
contractions). You can use `sum(contraction_cost(A; sequence))` to get the total cost of the
contraction.

If no sequence is specified, left associative contraction is used, in other words the sequence
is equivalent to `[[[[1, 2], 3], 4], …]`.
"""
function contraction_cost(A; sequence=left_associative_contraction_sequence(A))
  pairwise_costs = Number[]
  _contraction_cost!(pairwise_costs, A, sequence)
  return pairwise_costs
end

function _contraction_cost!(pairwise_costs, A, sequence)
  inds1 = _contraction_cost!(pairwise_costs, A, sequence[1])
  inds2 = _contraction_cost!(pairwise_costs, A, sequence[2])
  return _pairwise_contraction_cost!(pairwise_costs, inds1, inds2)
end

_contraction_cost!(pairwise_costs, As, sequence::Integer) = As[sequence]

function _pairwise_contraction_cost!(pairwise_costs, A1, A2)
  cost = dim(union(A1, A2))
  push!(pairwise_costs, cost)
  return symdiff(A1, A2)
end
