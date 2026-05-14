using ITensors: QN, SerializedQN, SerializedQNSpace
using JLD2: JLD2

# `Vector{Pair{QN, Int}}` deliberately has no `JLD2.writeas` mapping: declaring one would
# cause JLD2 to record `Index{Vector{Pair{QN, Int}}}` and `SerializedIndex{SerializedQNSpace}`
# as sharing a parameter reference on disk, which then confuses the inverse-lookup on read
# and reconstructs `Index{SerializedQNSpace}` instead of `SerializedIndex{SerializedQNSpace}`.
# The `space` field of `SerializedIndex{SerializedQNSpace}` is concretely typed, so JLD2
# stores it inline and there is no need for a writeas at this level — the methods below are
# called explicitly from `index.jl`.

function JLD2.wconvert(::Type{SerializedQNSpace}, qnblocks::Vector{Pair{QN, Int}})
    version = 1
    qns = SerializedQN[JLD2.wconvert(SerializedQN, first(p)) for p in qnblocks]
    dims = Int[last(p) for p in qnblocks]
    return SerializedQNSpace(version, qns, dims)
end

function JLD2.rconvert(::Type{Vector{Pair{QN, Int}}}, s::SerializedQNSpace)
    return Pair.(QN[JLD2.rconvert(QN, sqn) for sqn in s.qns], s.dims)
end
