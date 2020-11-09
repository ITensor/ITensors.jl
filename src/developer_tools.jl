
"""
inspectQNITensor is a developer-level debugging tool 
to look at internals or properties of QNITensors
"""
function inspectQNITensor(T::ITensor,is::QNIndexSet)
  #@show T.store.blockoffsets
  #@show T.store.data
  println("Block fluxes:")
  for b in nzblocks(T)
    @show flux(T,b)
  end
end
inspectQNITensor(T::ITensor,is::IndexSet) = nothing
inspectQNITensor(T::ITensor) = inspectQNITensor(T,inds(T))

"""
    pause()

Pauses execution until a key (other than 'q') is pressed.
Entering 'q' exits the program. The `pause()` function
is useful for inspecting output of programs at certain
points while giving the option to continue.
"""
function pause()
  print(stdout, "(Paused) ")
  c = read(stdin, 1)
  c == UInt8[0x71] && exit(0)
  return
end

# Boolean for debugging
global itdebug = false
set_debug(val::Bool) = (global itdebug = val)
get_debug() = itdebug


function printnz(T::ITensor{N}) where {N}
  print("ITensor ord=$(order(T))")
  print(" \n", inds(T))
  println(" \n", typeof(store(T)))
  for ind in Iterators.product(ntuple(n->1:dim(inds(T)[n]),N)...)
    val = T[ind...]
    if abs(val) > 1E-10
      print("  $ind $val\n")
    end
  end
  println()
end


