# TODO: Define `Adapt` overloads for ITensor/Tensor types
cpu(x::Array) = x
cpu(x::CuArray) = adapt(Array, x)
cpu(x::TensorStorage) = setdata(x, cpu(data(x)))
cpu(x::Tensor) = setstorage(x, cpu(storage(x)))

cu(x::TensorStorage) = setdata(x, cu(data(x)))
cu(x::Tensor) = setstorage(x, cu(storage(x)))
