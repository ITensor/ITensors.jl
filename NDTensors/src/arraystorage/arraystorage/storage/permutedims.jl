function permutedims!(
  output_array::MatrixOrArrayStorage, array::MatrixOrArrayStorage, perm, f::Function
)
  output_array = permutedims!!(
    leaf_parenttype(output_array), output_array, leaf_parenttype(array), array, perm, f
  )
  return output_array
end
