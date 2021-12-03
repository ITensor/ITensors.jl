struct Backend{backend} end

Backend(b::Backend) = b
Backend(s::AbstractString) = Backend{Symbol(s)}()
Backend(s::Symbol) = Backend{s}()
Backend(s::Nothing) = Backend{s}()
backend(::Backend{N}) where {N} = N
Backend() = Backend{Symbol()}()

macro Backend_str(s)
  return Backend{Symbol(s)}
end

const current_backend = Ref{Union{Nothing,Backend}}(nothing)

visualize(::Backend{nothing}, args...; kwargs...) = nothing

set_backend!(::Nothing) = (current_backend[] = nothing)
function set_backend!(backend::Backend)
  original_backend = current_backend[]
  current_backend[] = backend
  return original_backend
end
set_backend!(backend::Union{Symbol,String}) = set_backend!(Backend(backend))

get_backend() = isnothing(current_backend[]) ? default_backend() : current_backend[]

function plot(::Backend{T}, args...; kwargs...) where {T}
  return error("plot not implemented for backend type $T.")
end
function draw_edge!(::Backend{T}, args...; kwargs...) where {T}
  return error("draw_edge! not implemented for backend type $T.")
end
function annotate!(::Backend{T}, args...; kwargs...) where {T}
  return error("annotate! not implemented for backend type $T.")
end

function translate_color(::Backend{T}, color) where {T}
  return error("translate_color not implemented for backend type $T and color $color")
end

point_to_line(v1, v2) = ([v1[1], v2[1]], [v1[2], v2[2]])
