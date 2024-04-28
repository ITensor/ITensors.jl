using ..ITensors: ITensors
function ITensors._getindex(na::AbstractNamedDimsArray, I::Pair...)
  return na[I...]
end
function ITensors._setindex!!(na::AbstractNamedDimsArray, value::Int64, I::Pair...)
  na[I...] = value
  return na
end
