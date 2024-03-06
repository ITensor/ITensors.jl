using .TypeParameterAccessors:
  TypeParameterAccessors, Position, parameter, parenttype, set_type_parameter

TypeParameterAccessors.position(::Type{<:Block}, ::typeof(Base.ndims)) = Position(1);
NDTensors.ndims(type::Type{<:Block}) = parameter(type, Base.ndims)
NDTensors.ndims(type::Type{<:Blocks}) = NDTensors.ndims(parameter(type, Position(1)))
NDTensors.ndims(type::Type{<:BlockOffset}) = NDTensors.ndims(parameter(type, Position(1)))
NDTensors.ndims(type::Type{<:BlockOffsets}) = NDTensors.ndims(parameter(type, Position(1)))

NDTensors.ndims(blocks::Blocks) = ndims(typeof(blocks))
NDTensors.ndims(boff::BlockOffset) = ndims(typeof(boff))
NDTensors.ndims(boffs::BlockOffsets) = ndims(typeof(boffs))

TypeParameterAccessors.position(::Type{<:BlockSparse}, ::typeof(eltype)) = Position(1)
TypeParameterAccessors.position(::Type{<:BlockSparse}, ::typeof(parenttype)) = Position(2)
TypeParameterAccessors.position(::Type{<:BlockSparse}, ::typeof(Base.ndims)) = Position(3)

function NDTensors.set_ndims(storagetype::Type{<:BlockSparse}, param::Int)
  return set_type_parameter(storagetype, Base.ndims, param)
end

# TODO: Implement as `fieldtype(storagetype, :data)`.
#datatype(::Type{<:BlockSparse{<:Any,DataT}}) where {DataT} = DataT

# TODO: Implement as `fieldtype(storagetype, :blockoffsets)`.
blockoffsetstype(storagetype::Type{<:BlockSparse}) = BlockOffsets{ndims(storagetype)}

## TODO use set_eltype
## how does this work properly if we aren't converting the eltype of 
## the parenttype
Base.real(::Type{BlockSparse{T}}) where {T} = BlockSparse{real(T)}

complex(::Type{BlockSparse{T}}) where {T} = BlockSparse{complex(T)}
