using ..NamedDimsArrays: AbstractNamedDimsArray, dimnames, named, unname
using ...TensorAlgebra: TensorAlgebra, blockedperms, contract, contract!

function TensorAlgebra.contract!(
  na_dest::AbstractNamedDimsArray,
  na1::AbstractNamedDimsArray,
  na2::AbstractNamedDimsArray,
  α::Number=true,
  β::Number=false,
)
  contract!(
    unname(na_dest),
    dimnames(na_dest),
    unname(na1),
    dimnames(na1),
    unname(na2),
    dimnames(na2),
    α,
    β,
  )
  return na_dest
end

function TensorAlgebra.contract(
  na1::AbstractNamedDimsArray, na2::AbstractNamedDimsArray, α::Number=true
)
  a_dest, dimnames_dest = contract(
    unname(na1), dimnames(na1), unname(na2), dimnames(na2), α
  )
  return named(a_dest, dimnames_dest)
end
