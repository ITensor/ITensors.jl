
unique_siteind(A::AbstractMPS, B::AbstractMPS, j::Integer) =
  siteinds(uniqueind, A, B, j)

unique_siteinds(A::AbstractMPS, B::AbstractMPS, j::Integer) =
  siteinds(uniqueinds, A, B, j)

unique_siteinds(A::AbstractMPS, B::AbstractMPS) =
  siteinds(uniqueind, A, B)

common_siteind(A::AbstractMPS, B::AbstractMPS, j::Integer) =
  siteinds(commonind, A, B, j)

common_siteinds(A::AbstractMPS, B::AbstractMPS, j::Integer) =
  siteinds(commoninds, A, B, j)

common_siteinds(A::AbstractMPS, B::AbstractMPS) =
  siteinds(commoninds, A, B)

firstsiteind(M::AbstractMPS, j::Integer; kwargs...) =
  siteind(first, M, j; kwargs...)

map_linkinds!(f::Function, M::AbstractMPS) = map!(f, linkinds, M)

map_linkinds(f::Function, M::AbstractMPS) = map(f, linkinds, M)

map_common_siteinds!(f::Function, M1::AbstractMPS, M2::AbstractMPS) =
  map!(f, siteinds, commoninds, M1, M2)

map_common_siteinds(f::Function, M1::AbstractMPS, M2::AbstractMPS) =
  map(f, siteinds, commoninds, M1, M2)

map_unique_siteinds!(f::Function, M1::AbstractMPS, M2::AbstractMPS) =
  map!(f, siteinds, uniqueinds, M1, M2)

map_unique_siteinds(f::Function, M1::AbstractMPS, M2::AbstractMPS) =
  map(f, siteinds, uniqueinds, M1, M2)

for fname in (:sim, :prime, :setprime, :noprime, :addtags, :removetags,
              :replacetags, :settags)
  @eval begin
    $(Symbol(fname, :_linkinds))(M::AbstractMPS, args...; kwargs...) =
      map(i -> $fname(i, args...; kwargs...), linkinds, M)
    $(Symbol(fname, :_linkinds!))(M::AbstractMPS, args...; kwargs...) =
      map!(i -> $fname(i, args...; kwargs...), linkinds, M)
      $(Symbol(fname, :_common_siteinds))(M1::AbstractMPS, M2::AbstractMPS, args...; kwargs...) =
       map(i -> $fname(i, args...; kwargs...), siteinds, commoninds, M1, M2)
     $(Symbol(fname, :_common_siteinds!))(M1::AbstractMPS, M2::AbstractMPS, args...; kwargs...) =
      map!(i -> $fname(i, args...; kwargs...), siteinds, commoninds, M1, M2)
    $(Symbol(fname, :_unique_siteinds))(M1::AbstractMPS, M2::AbstractMPS, args...; kwargs...) =
      map(i -> $fname(i, args...; kwargs...), siteinds, uniqueinds, M1, M2)
    $(Symbol(fname, :_unique_siteinds!))(M1::AbstractMPS, M2::AbstractMPS, args...; kwargs...) =
      map!(i -> $fname(i, args...; kwargs...), siteinds, uniqueinds, M1, M2)
  end
end

