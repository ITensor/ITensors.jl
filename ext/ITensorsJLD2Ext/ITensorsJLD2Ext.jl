module ITensorsJLD2Ext

using ITensors: ITensors, ITensor, Index, QN, QNVal, TagSet
using JLD2: JLD2

# Bridge ITensors's backend-agnostic `serialized_type` mapping into JLD2's `writeas`
# mechanism. Per-type declarations rather than a single abstract-type catch-all because
# these types don't share a single supertype.

JLD2.writeas(::Type{QNVal}) = ITensors.serialized_type(QNVal)
JLD2.writeas(::Type{QN}) = ITensors.serialized_type(QN)
JLD2.writeas(::Type{TagSet}) = ITensors.serialized_type(TagSet)
JLD2.writeas(::Type{T}) where {T <: Index} = ITensors.serialized_type(T)
JLD2.writeas(::Type{ITensor}) = ITensors.serialized_type(ITensor)

# `JLD2.wconvert` is left at its default, which delegates to `Base.convert`. The
# `Base.convert(::Type{SchemaT}, ::InMemoryT)` shims in ITensors forward to
# `ITensors.serialize_convert`, so the write path bottoms out in the same source-of-truth
# function as the manual one without any JLD2-specific code here.

# Permissive `rconvert` for the read direction: `s` is duck-typed so this method matches
# JLD2's `AbstractReconstructedType` (produced when an on-disk schema struct's field
# layout differs from the current code's definition) as well as the canonical
# `SerializedX` struct. Delegates to the source-of-truth read function in ITensors,
# which does the field-driven value-level transform without caring whether `s` is the
# concrete schema struct or a reconstructed placeholder.
JLD2.rconvert(::Type{QNVal}, s) = ITensors.deserialize_convert(QNVal, s)
JLD2.rconvert(::Type{QN}, s) = ITensors.deserialize_convert(QN, s)
JLD2.rconvert(::Type{TagSet}, s) = ITensors.deserialize_convert(TagSet, s)
JLD2.rconvert(::Type{T}, s) where {T <: Index} = ITensors.deserialize_convert(T, s)
JLD2.rconvert(::Type{ITensor}, s) = ITensors.deserialize_convert(ITensor, s)

# Idempotency overloads — workaround for a JLD2 bug where `rconvert` is called twice for
# types with custom serialization that appear as fields inside another compound type
# (e.g. inside `Pair{QN, Int}`). When that second call lands here with the already-
# converted in-memory value, return it unchanged rather than re-running the read
# transform on something that isn't a serialized representation.
# TODO: Remove once the JLD2 double-`rconvert` bug is fixed.
JLD2.rconvert(::Type{QNVal}, x::QNVal) = x
JLD2.rconvert(::Type{QN}, x::QN) = x
JLD2.rconvert(::Type{TagSet}, x::TagSet) = x
JLD2.rconvert(::Type{T}, x::T) where {T <: Index} = x
JLD2.rconvert(::Type{ITensor}, x::ITensor) = x

end # module ITensorsJLD2Ext
