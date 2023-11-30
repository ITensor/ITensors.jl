# using ..ITensors: IndexID
using LinearAlgebra: LinearAlgebra, qr
using ...NDTensors.NamedDimsArrays: AbstractNamedDimsArray, dimnames, name, randname, unname
function LinearAlgebra.qr(na::AbstractNamedDimsArray; positive=nothing)
  # TODO: Make this more systematic.
  names_a = dimnames(na)
  # TODO: Create a `TensorAlgebra.qr`.
  q, r = qr(unname(na))
  # TODO: Use `sim` or `rand(::IndexID)`.
  name_qr = randname(names_a[1]) # IndexID(rand(UInt64), "", 0)
  # TODO: Make this GPU-friendly.
  nq = named(Matrix(q), (names_a[1], name_qr))
  nr = named(Matrix(r), (name_qr, names_a[2]))
  return nq, nr
end
