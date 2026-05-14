# On-disk struct layouts for ITensors core types. Used by ITensorsJLD2Ext (and intended
# to be reusable by other serialization backends in the future). Kept in the main package
# so that the type names recorded in serialized files do not encode the extension
# module's namespace.

struct SerializedQNVal
    version::Int
    name::String
    val::Int
    modulus::Int
end

struct SerializedQN
    version::Int
    qnvals::Vector{SerializedQNVal}
end

struct SerializedTagSet
    version::Int
    tags::Vector{String}
end

struct SerializedQNSpace
    version::Int
    qns::Vector{SerializedQN}
    dims::Vector{Int}
end

struct SerializedIndex{Space}
    version::Int
    id::UInt64
    space::Space
    dir::Int
    tags::SerializedTagSet
    plev::Int
end

struct SerializedITensor
    version::Int
    storage::Any
    inds::Vector
end
