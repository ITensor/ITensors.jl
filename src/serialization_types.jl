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

struct SerializedQNVal
    version::UInt32
    name::String
    val::Int64
    modulus::Int64
end

struct SerializedQN
    version::UInt32
    qnvals::Vector{SerializedQNVal}
end

struct SerializedTagSet
    version::UInt32
    tags::Vector{String}
end

struct SerializedQNSpace
    version::UInt32
    qns::Vector{SerializedQN}
    dims::Vector{Int64}
end

struct SerializedIndex{Space}
    version::UInt32
    id::UInt64
    space::Space
    dir::Int8
    tags::SerializedTagSet
    plev::Int64
end

struct SerializedITensor
    version::UInt32
    storage::Any
    inds::Vector
end
