function output_labels(
  f::typeof(contract),
  alg::Algorithm,
  a1::AbstractArray,
  labels1,
  a2::AbstractArray,
  labels2,
  α,
  β,
)
  return output_labels(f, alg, labels1, labels2)
end

function output_labels(f::typeof(contract), alg::Algorithm, labels1, labels2)
  return output_labels(f, labels1, labels2)
end

function output_labels(::typeof(contract), labels1, labels2)
  return symdiff(labels1, labels2)
end

function bipartitioned_permutations(
  f::typeof(contract), alg::Algorithm, labels_dest, labels1, labels2
)
  return bipartitioned_permutations(f, labels_dest, labels1, labels2)
end

function bipartitioned_permutations(::typeof(contract), labels_dest, labels1, labels2)
  labels12 = (labels1..., labels2...)
  if isodd(length(labels12) - length(labels_dest))
    error("Can't contract $labels1 and $labels2 to $labels_dest")
  end
  labels_contracted = unique(setdiff(labels12, labels_dest))
  labels1_uncontracted = setdiff(labels1, labels_contracted)
  labels2_uncontracted = setdiff(labels2, labels_contracted)
  # Positions of labels.
  pos_dest_1 = map(l -> findfirst(isequal(l), labels_dest), labels1_uncontracted)
  pos_dest_2 = map(l -> findfirst(isequal(l), labels_dest), labels2_uncontracted)
  pos1_contracted = map(l -> findfirst(isequal(l), labels1), labels_contracted)
  pos2_contracted = map(l -> findfirst(isequal(l), labels2), labels_contracted)
  pos1_uncontracted = map(l -> findfirst(isequal(l), labels1), labels1_uncontracted)
  pos2_uncontracted = map(l -> findfirst(isequal(l), labels2), labels2_uncontracted)
  # Bipartitioned permutations.
  biperm_dest = BipartitionedPermutation(pos_dest_1, pos_dest_2)
  biperm1 = BipartitionedPermutation(pos1_uncontracted, pos1_contracted)
  biperm2 = BipartitionedPermutation(pos2_contracted, pos2_uncontracted)
  return biperm_dest, biperm1, biperm2
end
