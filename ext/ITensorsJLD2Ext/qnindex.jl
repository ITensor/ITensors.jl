using ITensors: QN
using JLD2: JLD2

# Lives separately from SerializedIndex so that an `Index` with `space::Vector{Pair{QN,Int}}`
# (i.e. a QNIndex) can carry a concrete `SerializedQNSpace` as the `space` parameter rather
# than fall back to `Any`.

struct SerializedQNSpace
    version::Int
    qns::Vector{SerializedQN}
    dims::Vector{Int}
end

_wconvert_space(sp::Int) = sp

function _wconvert_space(qnblocks::Vector{Pair{QN, Int}})
    version = 1
    qns = SerializedQN[JLD2.wconvert(SerializedQN, first(p)) for p in qnblocks]
    dims = Int[last(qnblock) for qnblock in qnblocks]
    return SerializedQNSpace(version, qns, dims)
end

_rconvert_space(sp::Int) = sp

function _rconvert_space(sp)
    return Pair.(QN[JLD2.rconvert(QN, sqn) for sqn in sp.qns], sp.dims)
end
