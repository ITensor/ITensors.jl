for Typ in (:UnallocatedFill, :UnallocatedZeros)
  ## Here are functions specifically defined for UnallocatedArrays
  ## not implemented by FillArrays
  @eval begin
    alloctype(A::$Typ) = alloctype(typeof(A))
    alloctype(::Type{<:$Typ{ElT,N,Axes,Alloc}}) where {ElT,N,Axes,Alloc} = Alloc

    get_index(A::$Typ) = getindex(data(A))

    array(A::$Typ) = alloctype(typeof(A))(data(A))
    Array(A::$Typ) = array(A)
    ## A function for NDTensors to launch functions off of
    is_immutable(A::$Typ) = is_immutable(typeof(A))
    is_immutable(::Type{<:$Typ}) = true

  end
  ## TODO I don't think this is the correct way to call
  ## functions which are defined in `FillArrays`
  for fun in (:size, :length, :convert, :sum, :getindex_value)
    @eval FillArrays.$fun(A::$Typ) = $fun(data(A))
  end
  ## TODO Here I am defining LinearAlgebra functions in one sweep
  for fun in (:norm,)
    @eval LinearAlgebra.$fun(A::$Typ) = $fun(data(A))
  end

end

copy(F::UnallocatedFill) = UnallocatedFill{eltype(F), ndims(F), axes(F), alloctype(F)}(data(F)[1], size(F))