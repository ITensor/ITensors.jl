# Backend-agnostic serialization layer for ITensors core types.
#
# Pairs each in-memory type with its [`SerializedX`](@ref) schema struct via
# [`serialized_type`](@ref), and defines `Base.convert` overloads to do the value-level
# transform in both directions. Serialization backends (today only `ITensorsJLD2Ext`)
# layer on top: they map their own writeas mechanism to `serialized_type`, and the
# `Base.convert` overloads here handle the actual byte-shape transform.

"""
    ITensors.serialized_type(::Type{T}) -> Type

Return the on-disk schema struct type that an instance of `T` is serialized as.
Backend-agnostic — used by serialization extensions (e.g. `ITensorsJLD2Ext`) to bridge
into their own writeas / type-mapping mechanism.

```julia
ITensors.serialized_type(QN)         # => ITensors.SerializedQN
ITensors.serialized_type(Index{Int}) # => ITensors.SerializedIndex{Int}
```
"""
function serialized_type end

serialized_type(::Type{QNVal}) = SerializedQNVal
serialized_type(::Type{QN}) = SerializedQN
serialized_type(::Type{<:TagSet}) = SerializedTagSet
serialized_type(::Type{<:Index{S}}) where {S} = SerializedIndex{_serialized_space_type(S)}
serialized_type(::Type{ITensor}) = SerializedITensor

# Internal mapping from the in-memory `Index.space` type to the on-disk space type.
# Intentionally NOT registered as `serialized_type` itself: doing so would expose a
# JLD2-visible type mapping for `Vector{Pair{QN, Int}}` → `SerializedQNSpace`, which
# causes JLD2 to record `Index` and `SerializedIndex` as sharing a parameter ref on disk
# and crosses them on read. Handled here as a private helper instead.
_serialized_space_type(::Type{Int}) = Int
_serialized_space_type(::Type{Vector{Pair{QN, Int}}}) = SerializedQNSpace

# QNVal

function Base.convert(::Type{SerializedQNVal}, qv::QNVal)
    return SerializedQNVal(UInt32(1), String(name(qv)), val(qv), modulus(qv))
end

Base.convert(::Type{QNVal}, s::SerializedQNVal) = QNVal(s.name, s.val, s.modulus)

# QN

function Base.convert(::Type{SerializedQN}, qn::QN)
    qnvals = SerializedQNVal[convert(SerializedQNVal, qn[i]) for i in 1:nactive(qn)]
    return SerializedQN(UInt32(1), qnvals)
end

function Base.convert(::Type{QN}, s::SerializedQN)
    return QN([(qv.name, qv.val, qv.modulus) for qv in s.qnvals]...)
end

# TagSet

function Base.convert(::Type{SerializedTagSet}, ts::TagSet)
    return SerializedTagSet(UInt32(1), String[String(ts[n]) for n in 1:length(ts)])
end

Base.convert(::Type{TagSet}, s::SerializedTagSet) = TagSet(join(s.tags, ","))

# QN space (`Vector{Pair{QN, Int}}` in memory ↔ `SerializedQNSpace` on disk)

function Base.convert(::Type{SerializedQNSpace}, qnblocks::Vector{Pair{QN, Int}})
    qns = SerializedQN[convert(SerializedQN, first(p)) for p in qnblocks]
    dims = Int64[last(p) for p in qnblocks]
    return SerializedQNSpace(UInt32(1), qns, dims)
end

function Base.convert(::Type{Vector{Pair{QN, Int}}}, s::SerializedQNSpace)
    return Pair.(QN[convert(QN, sqn) for sqn in s.qns], s.dims)
end

# Index

function Base.convert(::Type{<:SerializedIndex}, i::Index)
    SP = _serialized_space_type(typeof(space(i)))
    sp = convert(SP, space(i))
    return SerializedIndex(
        UInt32(1), id(i), sp, Int8(dir(i)), convert(SerializedTagSet, tags(i)), plev(i)
    )
end

function Base.convert(::Type{I}, s::SerializedIndex) where {T, I <: Index{T}}
    sp = convert(T, s.space)
    return I(s.id, sp, Arrow(s.dir), convert(TagSet, s.tags), s.plev)
end

# ITensor

function Base.convert(::Type{SerializedITensor}, it::ITensor)
    return SerializedITensor(UInt32(1), storage(it), collect(inds(it)))
end

Base.convert(::Type{ITensor}, s::SerializedITensor) = itensor(s.storage, Tuple(s.inds))
