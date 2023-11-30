using NDTensors.NamedDimsArrays: AbstractNamedDimsArray, dimnames, named, unname
using NDTensors.TensorAlgebra: TensorAlgebra, contract

function TensorAlgebra.contract(
  na1::AbstractNamedDimsArray, na2::AbstractNamedDimsArray, α, β; kwargs...
)
  a_dest, names_dest = contract(
    unname(na1), dimnames(na1), unname(na2), dimnames(na2), α, β; kwargs...
  )
  # TODO: Automate `Tuple` conversion of names?
  return named(a_dest, Tuple(names_dest))
end

function TensorAlgebra.contract(
  na1::AbstractNamedDimsArray, na2::AbstractNamedDimsArray; kwargs...
)
  return contract(na1, na2, true, false; kwargs...)
end
