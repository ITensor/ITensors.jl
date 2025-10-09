using HDF5: HDF5, attributes, create_group, open_group, read, write
using NDTensors: Dense

function HDF5.write(
        parent::Union{HDF5.File, HDF5.Group}, name::String, D::Store
    ) where {Store <: Dense}
    g = create_group(parent, name)
    attributes(g)["type"] = "Dense{$(eltype(Store))}"
    attributes(g)["version"] = 1
    return if eltype(D) != Nothing
        write(g, "data", D.data)
    end
end

function HDF5.read(
        parent::Union{HDF5.File, HDF5.Group}, name::AbstractString, ::Type{Store}
    ) where {Store <: Dense}
    g = open_group(parent, name)
    ElT = eltype(Store)
    typestr = "Dense{$ElT}"
    if read(attributes(g)["type"]) != typestr
        error("HDF5 group or file does not contain $typestr data")
    end
    if ElT == Nothing
        return Dense{Nothing}()
    end
    # Attribute __complex__ is attached to the "data" dataset
    # by the h5 library used by C++ version of ITensor:
    if haskey(attributes(g["data"]), "__complex__")
        M = read(g, "data")
        nelt = size(M, 1) * size(M, 2)
        data = Vector(reinterpret(ComplexF64, reshape(M, nelt)))
    else
        data = read(g, "data")
    end
    return Dense{ElT}(data)
end
