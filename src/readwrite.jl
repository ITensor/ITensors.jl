
function readcpp(io::IO, ::Type{Vector{T}}; kwargs...) where {T}
  format = get(kwargs, :format, "v3")
  v = Vector{T}()
  if format == "v3"
    size = read(io, UInt64)
    resize!(v, size)
    for n in 1:size
      v[n] = readcpp(io, T; kwargs...)
    end
  else
    throw(ArgumentError("read Vector: format=$format not supported"))
  end
  return v
end
