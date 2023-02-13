adapt_structure(to, x::TensorStorage) = setdata(x, adapt(to, data(x)))
adapt_structure(to, x::Tensor) = setstorage(x, adapt(to, storage(x)))

cpu(eltype::Type{<:Number}, x) = fmap(x -> adapt(Array{eltype}, x), x)
cpu(x) = fmap(x -> adapt(Array, x), x)

# Implemented in `ITensorGPU`
function cu end

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

function adapt_storagetype(to::Type{<:AbstractArray}, x::Type{<:TensorStorage})
  return set_datatype(x, set_eltype_if_unspecified(to_vector_type(to), eltype(x)))
end

to_vector_type(arraytype::Type{<:AbstractVector}) = arraytype

to_vector_type(arraytype::Type{Array}) = Vector
to_vector_type(arraytype::Type{Array{T}}) where {T} = Vector{T}

function set_eltype_if_unspecified(
  arraytype::Type{<:AbstractArray{T}}, eltype::Type=default_eltype()
) where {T}
  return arraytype
end

#TODO transition to set_eltype when working for wrapped types
function set_eltype_if_unspecified(
  arraytype::Type{<:AbstractArray}, eltype::Type=default_eltype()
)
  return similartype(arraytype, eltype)
end
