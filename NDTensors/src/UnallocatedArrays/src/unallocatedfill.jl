## TODO All constructors not fully implemented but so far it matches the 
## constructors found in `FillArrays`. Need to fix io
struct UnallocatedFill{ElT,N,Axes,Alloc<:AbstractArray} <:
       FillArrays.AbstractFill{ElT,N,Axes}
  f::FillArrays.Fill{ElT,N,Axes}

  function UnallocatedFill{ElT,N,Axes,Alloc}(x::ElT, inds::Tuple) where {ElT,N,Axes,Alloc}
    f = FillArrays.Fill(x, inds)
    ax = typeof(FillArrays.axes(f))
    return new{ElT,N,ax,Alloc}(f)
  end
end

data(F::UnallocatedFill) = F.f
