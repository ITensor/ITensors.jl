function permutedims!(
  output_array::MatrixOrArrayStorage, array::MatrixOrArrayStorage, perm, f::Function
)
  output_array = permutedims!!(output_array, array, perm, f
  )
  return output_array
end
