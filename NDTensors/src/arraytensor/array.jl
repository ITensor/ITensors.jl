# Combiner
promote_rule(::Type{<:Combiner}, arraytype::Type{<:MatrixOrArrayStorage}) = arraytype

# Generic AbstractArray code
function contract(
  array1::MatrixOrArrayStorage,
  labels1,
  array2::MatrixOrArrayStorage,
  labels2,
  labelsR=contract_labels(labels1, labels2),
)
  output_array = contraction_output(array1, labels1, array2, labels2, labelsR)
  contract!(output_array, labelsR, array1, labels1, array2, labels2)
  return output_array
end

function contraction_output(
  array1::MatrixOrArrayStorage, array2::MatrixOrArrayStorage, indsR
)
  arraytypeR = contraction_output_type(typeof(array1), typeof(array2), indsR)
  return NDTensors.similar(arraytypeR, indsR)
end

function contraction_output_type(
  arraytype1::Type{<:MatrixOrArrayStorage}, arraytype2::Type{<:MatrixOrArrayStorage}, inds
)
  return similartype(promote_type(arraytype1, arraytype2), inds)
end

function contraction_output(
  array1::MatrixOrArrayStorage,
  labelsarray1,
  array2::MatrixOrArrayStorage,
  labelsarray2,
  labelsoutput_array,
)
  # TODO: Maybe use `axes` here to be more generic, for example for BlockArrays?
  indsoutput_array = contract_inds(
    size(array1), labelsarray1, size(array2), labelsarray2, labelsoutput_array
  )
  output_array = contraction_output(array1, array2, indsoutput_array)
  return output_array
end

# Required interface for specific AbstractArray types
function contract!(
  arrayR::MatrixOrArrayStorage,
  labelsR,
  array1::MatrixOrArrayStorage,
  labels1,
  array2::MatrixOrArrayStorage,
  labels2,
)
  props = ContractionProperties(labels1, labels2, labelsR)
  compute_contraction_properties!(props, array1, array2, arrayR)
  # TODO: Change this to just `contract!`, or maybe `contract_ttgt!`?
  _contract!(arrayR, array1, array2, props)
  return arrayR
end

function permutedims!(
  output_array::MatrixOrArrayStorage, array::MatrixOrArrayStorage, perm, f::Function
)
  output_array .= f.(output_array, permutedims!!(leaf_parenttype(array),array, perm))
  return output_array
end
