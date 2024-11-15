using ..NamedDimsArrays: AbstractNamedDimsArray, dimnames, named, unname
using ...SparseArraysBase: SparseArraysBase, densearray

# TODO: Use `Adapt` or some kind of rewrap function like in
# ArrayInterface.jl (https://github.com/JuliaArrays/ArrayInterface.jl/issues/136)
function SparseArraysBase.densearray(na::AbstractNamedDimsArray)
  return named(densearray(unname(na)), dimnames(na))
end
