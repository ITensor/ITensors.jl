using .TypeParameterAccessors: TypeParameterAccessors, specify_type_parameters
adapt_structure(to, x::TensorStorage) = setdata(x, adapt(to, data(x)))
adapt_structure(to, x::Tensor) = setstorage(x, adapt(to, storage(x)))

## use unwrap cpu here because Unwrap is included before NDTensors
Unwrap.cpu(eltype::Type{<:Number}, x) = fmap(x -> adapt(Array{eltype}, x), x)
Unwrap.cpu(x) = fmap(x -> adapt(Array, x), x)

# Implemented in `ITensorGPU` and NDTensorCUDA
function cu end

function mtl end

adapt_structure(to::Type{<:Number}, x::TensorStorage) = setdata(x, convert.(to, data(x)))

convert_scalartype(eltype::Type{<:Number}, x) = fmap(x -> adapt(eltype, x), x)

single_precision(::Type{Float32}) = Float32
single_precision(::Type{Float64}) = Float32
single_precision(eltype::Type{<:Complex}) = Complex{single_precision(real(eltype))}

single_precision(x) = fmap(x -> adapt(single_precision(eltype(x)), x), x)

double_precision(::Type{Float32}) = Float64
double_precision(::Type{Float64}) = Float64
double_precision(eltype::Type{<:Complex}) = Complex{double_precision(real(eltype))}

double_precision(x) = fmap(x -> adapt(double_precision(eltype(x)), x), x)

#
# Used to adapt `EmptyStorage` types
#

function adapt_storagetype(to::Type{<:AbstractVector}, x::Type{<:TensorStorage})
  return set_datatype(x, specify_type_parameters(to, eltype, eltype(x)))
end

## TODO refactor this
# function adapt_storagetype(to::Type{<:AbstractArray}, x::Type{<:TensorStorage})
#   return set_datatype(x, specify_parameters(set_ndims(to, 1), eltype(x)))
# end
