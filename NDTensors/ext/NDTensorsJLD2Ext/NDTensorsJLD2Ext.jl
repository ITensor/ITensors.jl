module NDTensorsJLD2Ext

using JLD2: JLD2
using NDTensors: NDTensors, TensorStorage

# Bridge NDTensors's backend-agnostic `serialized_type` mapping into JLD2's `writeas`
# mechanism, scoped to the `TensorStorage` hierarchy so JLD2's defaults for other types
# are untouched. The value-level conversions live as `Base.convert` overloads in NDTensors
# proper; JLD2's `wconvert` / `rconvert` already delegate to `Base.convert` by default.
JLD2.writeas(::Type{T}) where {T <: TensorStorage} = NDTensors.serialized_type(T)

end # module NDTensorsJLD2Ext
