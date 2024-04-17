using HDF5: HDF5, attributes, create_group, open_group, read, write
using ITensors: ITensor
using ITensors.ITensorMPS: MPO

function HDF5.write(parent::Union{HDF5.File,HDF5.Group}, name::AbstractString, M::MPO)
  g = create_group(parent, name)
  attributes(g)["type"] = "MPO"
  attributes(g)["version"] = 1
  N = length(M)
  write(g, "rlim", M.rlim)
  write(g, "llim", M.llim)
  write(g, "length", N)
  for n in 1:N
    write(g, "MPO[$(n)]", M[n])
  end
end

function HDF5.read(parent::Union{HDF5.File,HDF5.Group}, name::AbstractString, ::Type{MPO})
  g = open_group(parent, name)
  if read(attributes(g)["type"]) != "MPO"
    error("HDF5 group or file does not contain MPO data")
  end
  N = read(g, "length")
  rlim = read(g, "rlim")
  llim = read(g, "llim")
  v = [read(g, "MPO[$(i)]", ITensor) for i in 1:N]
  return MPO(v, llim, rlim)
end
