# Selected interface functions from https://github.com/ITensor/DiagonalArrays.jl,
# copied here so we don't have to depend on `DiagonalArrays.jl`.

function diaglength(a::AbstractArray)
    return minimum(size(a))
end

function diagstride(a::AbstractArray)
    s = 1
    p = 1
    for i in 1:(ndims(a) - 1)
        p *= size(a, i)
        s += p
    end
    return s
end

function diagindices(a::AbstractArray)
    maxdiag = if isempty(a)
        0
    else
        LinearIndices(a)[CartesianIndex(ntuple(Returns(diaglength(a)), ndims(a)))]
    end
    return 1:diagstride(a):maxdiag
end

function diagindices(a::AbstractArray{<:Any, 0})
    return Base.OneTo(1)
end

function diagview(a::AbstractArray)
    return @view a[diagindices(a)]
end
