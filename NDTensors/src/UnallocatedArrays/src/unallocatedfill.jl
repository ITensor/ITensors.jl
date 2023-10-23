## TODO All constructors not fully implemented but so far it matches the 
## constructors found in `FillArrays`. Need to fix io
struct UnallocatedFill{ElT,N,Axes,Alloc<:AbstractArray} <: FillArrays.AbstractFill{ElT,N,Axes} 
  f::FillArrays.Fill{ElT, N, Axes}

  function UnallocatedFill{ElT,N,Axes,Alloc}(x::ElT,inds::Tuple)where{ElT,N,Axes,Alloc}
    f = FillArrays.Fill(x, inds)
    ax = typeof(FillArrays.axes(f))
    new{ElT,N,ax,Alloc}(f)
  end

  function UnallocatedFill{ElT,0,Tuple{},Alloc}(x::ElT, inds::Tuple{}) where{ElT,Alloc}
    f = FillArrays.Fill{ElT,0,Tuple{}}(x,inds)
    new{ElT,0,Tuple{},Alloc}(f)
  end
end

function UnallocatedFill{ElT, Alloc}(x::ElT, inds::Tuple) where {ElT<:Number, Alloc<:AbstractArray}
  N = length(inds)
  Ax = Base.axes(inds)
  return UnallocatedFill{ElT, N, Ax,Alloc}(x, inds)
end

alloctype(::UnallocatedFill{ElT,N,Axes,Alloc}) where {ElT,N,Axes,Alloc} = Alloc
alloctype(::Type{<:UnallocatedFill{ElT,N,Axes,Alloc}}) where {ElT,N,Axes,Alloc} = Alloc

Base.axes(F::UnallocatedFill) = Base.axes(F.f)
Base.size(F::UnallocatedFill) = Base.size(F.f)
Base.length(F::UnallocatedFill) = Base.length(F.f)

Base.print_array(io::IO, X::UnallocatedFill) = Base.print_array(io, X.f)
