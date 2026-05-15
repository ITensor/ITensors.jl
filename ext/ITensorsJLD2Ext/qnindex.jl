using ITensors: QN, SerializedQN, SerializedQNSpace
using JLD2: JLD2

# Helpers for the `space` field of `Index` / `QNIndex`. Live in the extension because they
# call `JLD2.wconvert` / `JLD2.rconvert`; the struct itself (`SerializedQNSpace`) lives in
# `ITensors` proper.

_writeas_space(::Type{Int}) = Int
_writeas_space(::Type{Vector{Pair{QN, Int}}}) = SerializedQNSpace

_wconvert_space(sp::Int) = sp

function _wconvert_space(qnblocks::Vector{Pair{QN, Int}})
    version = UInt32(1)
    qns = SerializedQN[JLD2.wconvert(SerializedQN, first(p)) for p in qnblocks]
    dims = Int64[last(p) for p in qnblocks]
    return SerializedQNSpace(version, qns, dims)
end

_rconvert_space(sp::Int) = sp

function _rconvert_space(sp::SerializedQNSpace)
    return Pair.(QN[JLD2.rconvert(QN, sqn) for sqn in sp.qns], sp.dims)
end
