using ITensors

i = Index(200, "i")
A = randomITensor(i', dag(i))

#
# 2 tensors
#

println("\n2 tensors")

# Simple pair contraction
@btime $A' * $A

#
# 3 tensors
#

println("\n3 tensors")

# Pick sequence explicitly
@btime $A'' * ($A' * $A)

# Let it optimize
@btime $A'' * $A' * $A
@btime $A'' * $A * $A'

#
# 4 tensors
#

println("\n4 tensors")

# Pick sequence explicitly
@btime $A''' * ($A'' * ($A' * $A))

# Let it optimize
@btime $A''' * $A'' * $A' * $A
@btime $A'' * $A * $A''' * $A'

#
# 5 tensors
#

println("\n5 tensors")

# Pick sequence explicitly
@btime $A'''' * ($A''' * ($A'' * ($A' * $A)))

# Let it optimize
@btime $A'''' * $A''' * $A'' * $A' * $A
@btime $A'' * $A'''' * $A * $A''' * $A'


