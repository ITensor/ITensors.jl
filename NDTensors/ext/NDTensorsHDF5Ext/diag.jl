using HDF5: HDF5, attributes, create_group, open_group, read, write
using NDTensors: datatype, Dense, Diag

function HDF5.write(
        parent::Union{HDF5.File, HDF5.Group}, name::String, D::Store
    ) where {Store <: Diag}
    g = create_group(parent, name)
    attributes(g)["type"] = "Diag{$(eltype(Store)),$(datatype(Store))}"
    attributes(g)["version"] = 1
    return if eltype(D) != Nothing
        write(g, "data", D.data)
    end
end

function HDF5.read(
        parent::Union{HDF5.File, HDF5.Group}, name::AbstractString, ::Type{Store}
    ) where {Store <: Diag}
    g = open_group(parent, name)
    ElT = eltype(Store)
    DataT = datatype(Store)
    typestr = "Diag{$ElT,$DataT}"
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
    return Diag{ElT, DataT}(data)
end
