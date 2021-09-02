
function readcpp(io::IO, ::Type{Vector{T}}; kwargs...) where {T}
  format = get(kwargs, :format, "v3")
  v = Vector{T}()
  if format == "v3"
    size = HDF5.read(io, UInt64)
    resize!(v, size)
    for n in 1:size
      v[n] = readcpp(io, T; kwargs...)
    end
  else
    throw(ArgumentError("read Vector: format=$format not supported"))
  end
  return v
end

function HDF5.read(
  parent::Union{HDF5.File,HDF5.Group}, name::AbstractString, ::Type{AutoType}
)
  type_string = try
    @suppress_err begin
      g = open_group(parent, name)
      type_attribute = attributes(g)["type"]
      HDF5.read(type_attribute)
    end
  catch
    # If the file doesn't have the format
    # expected of ITensor HDF5 files, return `nothing`
    return nothing
  end
  T = Core.eval(Main, Meta.parse(type_string))
  return HDF5.read(parent, name, T)
end

# Tries to automatically determine the type from the "type" attribute.
# If the file is doesn't have the structure expected of an ITensor HDF5 file,
# try to read it with a general HDF5 fallback.
function read(parent::Union{HDF5.File,HDF5.Group}, name::AbstractString)
  r = HDF5.read(parent, name, AutoType)
  isnothing(r) && return HDF5.read(parent, name)
  return r
end

function save(filename::AbstractString, args...)
  h5open(filename, "w") do file
    write(file, args...)
  end
end

function load(filename::AbstractString, args...)
  h5open(filename, "r") do file
    read(file, args...)
  end
end
