data_isa(t::Tensor, datatype::Type) = data_isa(storage(t), datatype)
data_isa(s::TensorStorage, datatype::Type) = data_isa(data(s), datatype)
data_isa(d::AbstractArray, datatype::Type) = d isa datatype # Might have to unwrap if it is reshaped, sliced, etc.