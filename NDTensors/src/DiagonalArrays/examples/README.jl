# # DiagonalArrays.jl
#
# A Julia `DiagonalArray` type.

using NDTensors.DiagonalArrays:
  DiagonalArray,
  DiagIndex,
  DiagIndices,
  densearray

d = DiagonalArray([1.0, 2, 3], 3, 4, 5)
@show d[1, 1, 1] == 1
@show d[2, 2, 2] == 2
@show d[1, 2, 1] == 0

d[2, 2, 2] = 22
@show d[2, 2, 2] == 22

@show length(d[DiagIndices()]) == 3
@show densearray(d) == d
@show d[DiagIndex(2)] == d[2, 2, 2]

d[DiagIndex(2)] = 222
@show d[2, 2, 2] == 222

a = randn(3, 4, 5)
new_diag = randn(3)
a[DiagIndices()] = new_diag
d[DiagIndices()] = a[DiagIndices()]

@show a[DiagIndices()] == new_diag
@show d[DiagIndices()] == new_diag

# You can generate this README with:
# ```julia
# using Literate
# Literate.markdown("examples/README.jl", "."; flavor=Literate.CommonMarkFlavor())
# ```
