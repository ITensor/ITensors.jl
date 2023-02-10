set_properties_if_unspecified(storetype::Type{<:Dense{ElT, DataT}}, datatype::Type{<:AbstractArray} = default_datatype(ElT)) where {ElT, DataT<:AbstractArray{ElT}} = storetype
set_properties_if_unspecified(storetype::Type{<:Dense{ElT, DataT}}, datatype::Type{<:AbstractArray} = default_datatype(ElT)) where {ElT, DataT<:AbstractArray} = set_datatype(storetype, set_properties_if_unspecified(DataT, ElT))
set_properties_if_unspecified(storetype::Type{<:Dense{ElT}},
 datatype::Type{<:AbstractArray} = default_datatype(ElT)) where {ElT} = 
 default_storagetype(set_properties_if_unspecified(datatype, ElT), ())

set_properties_if_unspecified(storetype::Type{<:Dense}, datatype::Type{<:AbstractArray} = default_datatype(default_eltype())) =
default_storagetype(datatype, ())