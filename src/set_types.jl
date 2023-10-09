import NDTensors:leaf_parenttype
NDTensors.leaf_parenttype(T::ITensor) = leaf_parenttype(NDTensors.datatype(T))
