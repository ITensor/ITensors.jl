# using ..ITensors: IndexID
using LinearAlgebra: LinearAlgebra, qr
using ...NDTensors.NamedDimsArrays: AbstractNamedDimsArray, dimnames, name, randname, unname

function LinearAlgebra.qr(na::AbstractNamedDimsArray; positive=nothing)
  @assert isnothing(positive)
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

function LinearAlgebra.qr(
  na::AbstractNamedDimsArray, labels_codomain::Tuple, labels_domain::Tuple; positive=nothing
)
  @assert isnothing(positive)
  q, r = qr(unname(na), dimnames(na), name.(labels_codomain), name.(labels_domain))
  name_qr = randname(dimnames(na)[1])
  dimnames_q = (name.(labels_codomain)..., name_qr)
  dimnames_r = (name_qr, name.(labels_domain)...)
  return named(q, dimnames_q), named(r, dimnames_r)
end
