# On-disk struct layouts for ITensors core types. Used by ITensorsJLD2Ext (and intended
# to be reusable by other serialization backends in the future). Kept in the main package
# so that the type names recorded in serialized files do not encode the extension
# module's namespace.
#
# Integer-width conventions for cross-language readability:
#   * `version::UInt32` matches `Base.VersionNumber`'s field width.
#   * `Int64` is used for any logically-signed quantity that could in principle be
#     negative (`QN` charge values, modulus sentinel, prime level).
#   * `Int8` for `Arrow` direction, with values `In=-1`, `Neither=0`, `Out=+1`.
#   * `UInt64` for `Index` identifiers.

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
