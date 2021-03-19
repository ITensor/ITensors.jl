
module ContractionSequenceOptimization

  using NDTensors

  import NDTensors:
    dim

  export optimized_contraction_sequence

  include("utils.jl")
  include("three_tensors.jl")
  include("breadth_first_constructive.jl")

  # The integer type of the dimensions and costs.
  # Needs to be large to avoid overflow.
  const DimT = UInt128

  """
      optimized_contraction_sequence(T)

  Returns a contraction sequence for contracting the tensors `T`. The sequence is generally optimal (currently, outer product contractions are skipped, but some optimal sequences require outer product contractions).
  """
  function optimized_contraction_sequence(T)
    if length(T) == 1
      return Any[1]
    elseif length(T) == 2
      return Any[1, 2]
    elseif length(T) == 3
      return optimized_contraction_sequence(T[1], T[2], T[3])
    end
    return breadth_first_constructive(T)
  end

end # module ContractionSequenceOptimization

