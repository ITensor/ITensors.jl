using ..NamedDimsArrays: AbstractNamedDimsArray, dimnames, unname
using ...TensorAlgebra: TensorAlgebra, blockedperms, contract, contract!

function TensorAlgebra.contract!(
  na_dest::AbstractNamedDimsArray,
  na1::AbstractNamedDimsArray,
  na2::AbstractNamedDimsArray,
  α=true,
  β=false,
)
  biperm_dest, biperm1, biperm2 = blockedperms(
    contract, dimnames(na_dest), dimnames(na1), dimnames(na2)
  )
  contract!(unname(na_dest), biperm_dest, unname(na1), biperm1, unname(na2), biperm2, α, β)
  return na_dest
end
