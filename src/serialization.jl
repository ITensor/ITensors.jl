# Backend-agnostic serialization layer for ITensors core types.
#
# Each section below defines, for one in-memory type:
#   1. The [`SerializedX`](@ref) schema struct (the on-disk layout).
#   2. The [`serialized_type`](@ref) declaration mapping the in-memory type to the
#      schema struct, used by serialization backends to bridge into their own writeas /
#      type-mapping mechanism.
#   3. `Base.convert` overloads for the value-level transform in both directions.
#
# JLD2's default `wconvert` / `rconvert` already delegate to `Base.convert`, so backends
# (today only `ITensorsJLD2Ext`) only need to register `JLD2.writeas` declarations
# pointing at `serialized_type` — no JLD2-specific value-conversion code is required.
#
# Integer-width conventions for cross-language readability:
#   * `version::UInt32` matches `Base.VersionNumber`'s field width.
#   * `Int64` is used for any logically-signed quantity that could in principle be
#     negative (`QN` charge values, modulus sentinel, prime level).
#   * `Int8` for `Arrow` direction, with values `In=-1`, `Neither=0`, `Out=+1`.
#   * `UInt64` for `Index` identifiers.

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

# --- QNVal ---

"""
    SerializedQNVal

On-disk schema for a `QNVal` (a single named charge of a `QN`). Version 1.

Fields:

  - `version::UInt32`
  - `name::String` — the charge name (e.g. `"Sz"`, `"Nf"`).
  - `val::Int64` — the charge value.
  - `modulus::Int64` — `0` for a `ℤ` charge, `> 0` for a `ℤ_N` charge, with the same
    sentinel convention as `QNVal` in memory.
"""
struct SerializedQNVal
    version::UInt32
    name::String
    val::Int64
    modulus::Int64
end

serialized_type(::Type{QNVal}) = SerializedQNVal

function Base.convert(::Type{SerializedQNVal}, qv::QNVal)
    return SerializedQNVal(UInt32(1), String(name(qv)), val(qv), modulus(qv))
end

Base.convert(::Type{QNVal}, s::SerializedQNVal) = QNVal(s.name, s.val, s.modulus)

# --- QN ---

"""
    SerializedQN

On-disk schema for a `QN`. Version 1. Stores only the active charges.

Fields:

  - `version::UInt32`
  - `qnvals::Vector{SerializedQNVal}`
"""
struct SerializedQN
    version::UInt32
    qnvals::Vector{SerializedQNVal}
end

serialized_type(::Type{QN}) = SerializedQN

function Base.convert(::Type{SerializedQN}, qn::QN)
    qnvals = SerializedQNVal[convert(SerializedQNVal, qn[i]) for i in 1:nactive(qn)]
    return SerializedQN(UInt32(1), qnvals)
end

function Base.convert(::Type{QN}, s::SerializedQN)
    return QN([(qv.name, qv.val, qv.modulus) for qv in s.qnvals]...)
end

# --- TagSet ---

"""
    SerializedTagSet

On-disk schema for a `TagSet`. Version 1.

Fields:

  - `version::UInt32`
  - `tags::Vector{String}` — one entry per active tag, in canonical order.
"""
struct SerializedTagSet
    version::UInt32
    tags::Vector{String}
end

serialized_type(::Type{<:TagSet}) = SerializedTagSet

function Base.convert(::Type{SerializedTagSet}, ts::TagSet)
    return SerializedTagSet(UInt32(1), String[String(ts[n]) for n in 1:length(ts)])
end

Base.convert(::Type{TagSet}, s::SerializedTagSet) = TagSet(join(s.tags, ","))

# --- QN space (the `space` field of a `QNIndex`) ---

"""
    SerializedQNSpace

On-disk schema for the `space` of a `QNIndex` (`Vector{Pair{QN, Int}}` in memory).
Version 1. Stored as parallel arrays so that a cross-language reader sees plain
`SerializedQN` and integer dimension lists rather than a nested pair structure.

Fields:

  - `version::UInt32`
  - `qns::Vector{SerializedQN}` — one entry per block.
  - `dims::Vector{Int64}` — same length as `qns`; the block dimensions.
"""
struct SerializedQNSpace
    version::UInt32
    qns::Vector{SerializedQN}
    dims::Vector{Int64}
end

# Note: `Vector{Pair{QN, Int}}` is deliberately NOT registered with `serialized_type`.
# Exposing that mapping would register a JLD2-visible writeas for the Vector type, which
# causes JLD2 to record `Index` and `SerializedIndex` as sharing a parameter ref on disk
# and crosses them on read. The `Vector{Pair{QN, Int}} <-> SerializedQNSpace` translation
# is invoked explicitly by the `Index` conversion below via the private helper
# `_serialized_space_type`.

function Base.convert(::Type{SerializedQNSpace}, qnblocks::Vector{Pair{QN, Int}})
    qns = SerializedQN[convert(SerializedQN, first(p)) for p in qnblocks]
    dims = Int64[last(p) for p in qnblocks]
    return SerializedQNSpace(UInt32(1), qns, dims)
end

function Base.convert(::Type{Vector{Pair{QN, Int}}}, s::SerializedQNSpace)
    return Pair.(QN[convert(QN, sqn) for sqn in s.qns], s.dims)
end

# --- Index ---

"""
    SerializedIndex{Space}

On-disk schema for an `Index`. Version 1. `Space` is the serialized space type:
`Int` for a non-QN `Index`, [`SerializedQNSpace`](@ref) for a `QNIndex`.

Fields:

  - `version::UInt32`
  - `id::UInt64`
  - `space::Space`
  - `dir::Int8` — `Arrow` direction: `In = -1`, `Neither = 0`, `Out = +1`.
  - `tags::SerializedTagSet`
  - `plev::Int64`
"""
struct SerializedIndex{Space}
    version::UInt32
    id::UInt64
    space::Space
    dir::Int8
    tags::SerializedTagSet
    plev::Int64
end

# Internal mapping from the in-memory `Index.space` type to the on-disk space type.
# Used by `serialized_type(::Type{<:Index})` and `Base.convert(::Type{<:SerializedIndex}, ::Index)`.
_serialized_space_type(::Type{Int}) = Int
_serialized_space_type(::Type{Vector{Pair{QN, Int}}}) = SerializedQNSpace

serialized_type(::Type{<:Index{S}}) where {S} = SerializedIndex{_serialized_space_type(S)}

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

# --- ITensor ---

"""
    SerializedITensor

On-disk schema for an `ITensor`. Version 1.

Fields:

  - `version::UInt32`
  - `storage::Any` — one of the `Serialized*` storage types from `NDTensors`
    ([`NDTensors.SerializedDense`](@ref), [`NDTensors.SerializedBlockSparse`](@ref),
    etc.). The concrete type is recorded by the backend (e.g. JLD2 stores it in the
    type table); cross-language readers dispatch on it.
  - `inds::Vector` — a list of [`SerializedIndex`](@ref) values.

# Note on shape

Conceptually this would be `SerializedITensor{Storage, Space}` with concrete
`storage::Storage` and `inds::Vector{SerializedIndex{Space}}` fields, matching the
Serialized-fields-of-Serialized-types convention used elsewhere in this file. That form
is blocked by `JLD2.writeas`'s type-level dispatch: it requires a single concrete target
whose fields it can introspect at type level, but the parameters depend on the runtime
storage / index-space types and `ITensor` itself is non-parametric, so `writeas` cannot
compute them and a `UnionAll` return errors. Revisit if JLD2 changes that design.
"""
struct SerializedITensor
    version::UInt32
    storage::Any
    inds::Vector
end

serialized_type(::Type{ITensor}) = SerializedITensor

function Base.convert(::Type{SerializedITensor}, it::ITensor)
    return SerializedITensor(UInt32(1), storage(it), collect(inds(it)))
end

Base.convert(::Type{ITensor}, s::SerializedITensor) = itensor(s.storage, Tuple(s.inds))
