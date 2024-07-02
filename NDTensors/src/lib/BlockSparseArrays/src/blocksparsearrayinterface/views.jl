function blocksparse_view(a, I...)
  return Base.invoke(view, Tuple{AbstractArray,Vararg{Any}}, a, I...)
end
