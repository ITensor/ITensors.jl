module ITensorsJLD2Ext

using ITensors: ITensors, ITensor, Index, QN, QNVal, TagSet
using JLD2: JLD2

# Bridge ITensors's backend-agnostic `serialized_type` mapping into JLD2's `writeas`
# mechanism. Per-type declarations rather than a single abstract-type catch-all because
# these types don't share a single supertype.
#
# The value-level conversions live as `Base.convert` overloads in ITensors proper; JLD2's
# `wconvert` / `rconvert` already delegate to `Base.convert` by default, so no
# JLD2-specific value-conversion code is needed here.

JLD2.writeas(::Type{QNVal}) = ITensors.serialized_type(QNVal)
JLD2.writeas(::Type{QN}) = ITensors.serialized_type(QN)
JLD2.writeas(::Type{<:TagSet}) = ITensors.serialized_type(TagSet)
JLD2.writeas(::Type{T}) where {T <: Index} = ITensors.serialized_type(T)
JLD2.writeas(::Type{ITensor}) = ITensors.serialized_type(ITensor)

end # module ITensorsJLD2Ext
