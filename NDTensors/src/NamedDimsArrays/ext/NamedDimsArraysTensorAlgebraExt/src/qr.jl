# using ..ITensors: IndexID
using LinearAlgebra: LinearAlgebra, qr
using ...NDTensors.NamedDimsArrays: AbstractNamedDimsArray, dimnames, name, unname
function LinearAlgebra.qr(na::AbstractNamedDimsArray; positive=nothing)
  # TODO: Make this more systematic.
  i, j = dimnames(na)
  # TODO: Create a `TensorAlgebra.qr`.
  q, r = qr(unname(na))
  # TODO: Use `sim` or `rand(::IndexID)`.
  name_qr = IndexID(rand(UInt64), "", 0)
  # TODO: Make this GPU-friendly.
  nq = named(Matrix(q), (i, name_qr))
  nr = named(Matrix(r), (name_qr, j))
  return nq, nr
end
