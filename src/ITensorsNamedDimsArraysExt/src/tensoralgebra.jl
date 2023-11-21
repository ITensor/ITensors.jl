using ..NDTensors.NamedDimsArrays: AbstractNamedDimsArray, align
using ..NDTensors.TensorAlgebra: TensorAlgebra
using ..NDTensors: NeverAlias
using ..ITensors: ITensors
ITensors._contract(na1::AbstractNamedDimsArray, na2::AbstractNamedDimsArray) = TensorAlgebra.contract(na1, na2)
ITensors._add(na1::AbstractNamedDimsArray, na2::AbstractNamedDimsArray) = error("Not implemented yet")
ITensors._permute(::NeverAlias, na::AbstractNamedDimsArray, dims::Tuple) = align(na, dims)
