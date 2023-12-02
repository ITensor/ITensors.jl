using VectorInterface: zerovector!

##################################################
# TODO: Move to `DictionariesVectorInterfaceExt`.
using VectorInterface: VectorInterface, zerovector!, zerovector!!
using Dictionaries: AbstractDictionary

function VectorInterface.zerovector!(x::AbstractDictionary{<:Number})
  return fill!(x, zero(scalartype(x)))
end
function VectorInterface.zerovector!(x::AbstractDictionary)
  T = eltype(x)
  for I in eachindex(x)
    if isbitstype(T) || isassigned(x, I)
      x[I] = zerovector!!(x[I])
    else
      x[I] = zero(eltype(x))
    end
  end
  return x
end
##################################################

function sparse_zerovector!(a::AbstractArray)
  dropall!(a)
  zerovector!(sparse_storage(a))
  return a
end
