using HDF5: HDF5, attributes, create_group, open_group, read, write
using ITensors: Index, Indices

function HDF5.write(parent::Union{HDF5.File, HDF5.Group}, name::AbstractString, is::Indices)
    g = create_group(parent, name)
    attributes(g)["type"] = "IndexSet"
    attributes(g)["version"] = 1
    N = length(is)
    write(g, "length", N)
    for n in 1:N
        write(g, "index_$n", is[n])
    end
    return nothing
end

function HDF5.read(
        parent::Union{HDF5.File, HDF5.Group}, name::AbstractString, T::Type{<:Indices}
    )
    g = open_group(parent, name)
    if read(attributes(g)["type"]) != "IndexSet"
        error("HDF5 group or file does not contain IndexSet data")
    end
    n = read(g, "length")
    return T(Index[read(g, "index_$j", Index) for j in 1:n])
end
