

function Base.read(io::IO,::Type{Vector{T}};kwargs...) where {T}
  format = get(kwargs,:format,"hdf5")
  v = Vector{T}()
  if format=="cpp"
    size = read(io,UInt64)
    resize!(v,size)
    for n=1:size
      v[n] = read(io,T;kwargs...)
    end
  else
    throw(ArgumentError("read Vector: format=$format not supported"))
  end
  return v
end
