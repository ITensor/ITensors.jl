module NDTensorsJLD2Ext

using JLD2: JLD2
using NDTensors: NDTensors, TensorStorage

# Bridge NDTensors's backend-agnostic `serialized_type` mapping into JLD2's `writeas`
# mechanism, scoped to the `TensorStorage` hierarchy so JLD2's defaults for other types
# are untouched.
JLD2.writeas(::Type{T}) where {T <: TensorStorage} = NDTensors.serialized_type(T)

# `JLD2.wconvert` is left at its default, which delegates to `Base.convert`. The
# `Base.convert(::Type{SchemaT}, ::InMemoryT)` shims in NDTensors then forward to
# `NDTensors.serialize_convert`, so the write path bottoms out in the same source-of-truth
# function as the manual one without any JLD2-specific code here.

# Permissive `rconvert` for the read direction: `s` is duck-typed so this method matches
# JLD2's `AbstractReconstructedType` (produced when an on-disk schema struct's field
# layout differs from the current code's definition) as well as the canonical
# `SerializedX` struct. Delegates to the source-of-truth read function in NDTensors,
# which does the field-driven value-level transform without caring whether `s` is the
# concrete schema struct or a reconstructed placeholder.
JLD2.rconvert(::Type{T}, s) where {T <: TensorStorage} =
    NDTensors.deserialize_convert(T, s)

# Idempotency overload — workaround for a JLD2 bug where `rconvert` is called twice for
# types with custom serialization that appear as fields inside another compound type
# (e.g. inside a `Pair`). When that second call lands here with the already-converted
# in-memory value, return it unchanged rather than re-running the read transform on
# something that isn't a serialized representation.
# TODO: Remove once the JLD2 double-`rconvert` bug is fixed.
JLD2.rconvert(::Type{T}, x::T) where {T <: TensorStorage} = x

end # module NDTensorsJLD2Ext
