# TODO: Delete once `TensorStorage` is removed.
function NDTensors.to_arraystorage(x::ITensor)
  return itensor(NDTensors.to_arraystorage(tensor(x)))
end

# TODO: Delete once `TensorStorage` is removed.
function NDTensors.to_arraystorage(x::AbstractMPS)
  return NDTensors.to_arraystorage.(x)
end
