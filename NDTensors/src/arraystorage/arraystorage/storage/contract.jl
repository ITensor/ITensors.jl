# Generic AbstractArray code
function contract(
  array1::MatrixOrArrayStorage,
  labels1,
  array2::MatrixOrArrayStorage,
  labels2,
  labelsR=contract_labels(labels1, labels2);
  kwargs...
)
  output_array = contraction_output(array1, labels1, array2, labels2, labelsR)
  contract!(output_array, labelsR, array1, labels1, array2, labels2; kwargs...)
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
# TODO: Define `default_α` and `default_β`.
# TODO: Define this as a `ttgt` or `matricize` backend.
function contract!(
  array_dest::MatrixOrArrayStorage,
  labels_dest,
  array1::MatrixOrArrayStorage,
  labels1,
  array2::MatrixOrArrayStorage,
  labels2,
  α=one(eltype(array_dest)),
  β=zero(eltype(array_dest));
)
  props = ContractionProperties(labels1, labels2, labels_dest)
  compute_contraction_properties!(props, array1, array2, array_dest)
  # TODO: Change this to just `contract!`, or maybe `contract_ttgt!`?
  _contract!(array_dest, array1, array2, props, α, β)
  return array_dest
end
