
"""
inspectQNITensor is a developer-level debugging tool 
to look at internals or properties of QNITensors
"""
function inspectQNITensor(T::ITensor, is::QNIndexSet)
  #@show T.store.blockoffsets
  #@show T.store.data
  println("Block fluxes:")
  for b in nzblocks(T)
    @show flux(T, b)
  end
end
inspectQNITensor(T::ITensor, is::IndexSet) = nothing
inspectQNITensor(T::ITensor) = inspectQNITensor(T, inds(T))

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
  return nothing
end
