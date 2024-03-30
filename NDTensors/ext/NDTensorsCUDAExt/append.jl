## CUDA.jl has no append! function and 
## calling base results in a scalar issue
## This is probably super slow and I am seeing 
## very slow blocksparse DMRG but it does work.
function Base.append!(a::CuArray, b::CuArray)
  combined = vcat(a, b)
  resize!(a, length(combined))
  return CUDA.unsafe_copyto!(a, 1, combined, 1, length(combined))
end
