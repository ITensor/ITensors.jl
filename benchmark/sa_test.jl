using ITensors

a = SmallArray{Int}(1)
@show a[1]

b = SmallArray{Int}(1,2)
@show b[1]
@show b[2]

b[2] = 7
@show b

c = SmallArray{Int}([1,2,3])
@show c
@show c[2]
push!(c,10)
@show c

d = SmallArray{Int}([1,2,3,4])
@show d
@show d[2]
println("d = ")
for i in d
  println(i)
end

push!(d,10)
@show d
println("d = ")
for i in d
  println(i)
end

f = SmallArray{Int}([1,2,3,4,5])
@show f
@show f[2]
println("f = ")
for i in f
  println(i)
end

#is = SmallArray{Index}(Index(2),Index(3))
is = SmallArray{Index}(Index(1),Index(2),Index(3),Index(4),Index(5),Index(6))
@show is
println("is = ")
for i in is
  println(i)
end
