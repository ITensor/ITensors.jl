for Typ in (:UnallocatedFill, :UnallocatedZeros)
  ## Here are functions specifically defined for UnallocatedArrays
  ## not implemented by FillArrays
  ## TODO use set_parameters/get_parameters instead of alloctype and other 
  ## type info functions.
  ## TODO determine min number of functions needed to be forwarded
  @eval begin
    alloctype(A::$Typ) = alloctype(typeof(A))
    alloctype(Atype::Type{<:$Typ}) = get_parameter(Atype, Position{4}())

    allocate(A::$Typ) = alloctype(A)(parent(A))

    ## With these functions defined I can print UnallocatedArrays
    ## compute things like sum and norm, compute the size and length
    @inline Base.axes(A::$Typ) = axes(parent(A))
    Base.size(A::$Typ) = size(parent(A))
    FillArrays.getindex_value(A::$Typ) = FillArrays.getindex_value(parent(A))
    copy(A::$Typ) = A
    ## TODO Implement vec
    # Base.vec(Z::UnallocatedZeros) = typeof(Z)(length(Z))
    ## TODO Still working here I am not sure these functions and the
    ## Set parameter functions are working properly
    set_alloctype(F::Type{<:$Typ}, alloc::Type{<:AbstractArray}) =
      set_parameter(F, Position{4}(), alloc)

    Base.complex(A::$Typ) = set_alloctype(
      complex(parent(A)), set_eltype(alloctype(A), complex(eltype(alloctype(A))))
    )
  end
end
