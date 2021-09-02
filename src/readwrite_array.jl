ITensorHDF5Types = Union{TagSet,Index,ITensor}

hdf5_type(::Type{ITensor}) = ITensor
hdf5_type(::Type{<:Index}) = Index
hdf5_type(::Type{TagSet}) = TagSet
hdf5_type(::Type{<:AbstractArray{T}}) where {T} = Array{hdf5_type(T)}
hdf5_type(::T) where {T} = hdf5_type(T)

function HDF5.write(
  parent::Union{HDF5.File,HDF5.Group},
  name::AbstractString,
  d::AbstractArray{<:ITensorHDF5Types},
)
  g = create_group(parent, name)
  attributes(g)["type"] = "$(hdf5_type(d))"
  attributes(g)["version"] = 1
  write(g, "size", collect(size(d)))
  for c in CartesianIndices(d)
    write(g, "$(Tuple(c))", d[c])
  end
end

function HDF5.read(
  parent::Union{HDF5.File,HDF5.Group},
  name::AbstractString,
  T::Type{<:Array{<:ITensorHDF5Types}},
)
  g = open_group(parent, name)
  g_type = HDF5.read(attributes(g)["type"])
  if g_type != string(T)
    error(
      "HDF5 group or file does not contain $T data, instead it has data of type $g_type"
    )
  end
  g_size = HDF5.read(g, "size")
  eltype_T = eltype(T)
  return [HDF5.read(g, "$(Tuple(c))", eltype_T) for c in CartesianIndices(Tuple(g_size))]
end
