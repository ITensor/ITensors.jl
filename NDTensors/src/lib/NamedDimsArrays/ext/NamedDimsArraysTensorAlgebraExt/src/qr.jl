# using ..ITensors: IndexID
using LinearAlgebra: LinearAlgebra, qr
using ...NDTensors.NamedDimsArrays: AbstractNamedDimsArray, dimnames, name, randname, unname

function LinearAlgebra.qr(na::AbstractNamedDimsArray; positive=nothing)
  return qr(na, (dimnames(na, 1),), (dimnames(na, 2),); positive)
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
