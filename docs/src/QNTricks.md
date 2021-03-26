# Symmetric Tensor Background and Usage

Here is a collection of background material and example codes for understanding how symmetric tensors (tensors with conserved quantum numbers) work in ITensors.jl

## Combiners and Symmetric Tensors

In ITensors.jl, combiners are special sparse tensors that represent the action of taking the tensor product of one or more indices. It generalizes the idea of reshaping and permuting. For dense ITensors, a combiner is just the action of permuting and reshaping the data of the tensor. For symmetric tensors (quantum number conserving tensors represented as block sparse tensors), the combiner also fuses symmetry sectors together. They can be used for various purposes. Generally they are used internally in the library, for example in order to reshape a high order ITensor into an order 2 ITensor to perform a matrix decomposition like an SVD or eigendecomposition.

For example:
```@repl
using ITensors

# This is a short code showing how a combiner
# can be used to "flip" the direction of an Index
i = Index([QN(0) => 2, QN(1) => 3], "i")
j = Index([QN(0) => 2, QN(1) => 3], "j")
A = randomITensor(i, dag(j))
C = combiner(i, dag(j); tags = "c", dir = dir(i))
inds(A)
inds(A * C)
```
You can see that the combiner reshapes the indices of `A` into a single Index that contains the tensor product of the two input spaces. The spaces have size `QN(-1) => 2 * 3`, `QN(0) => 2 * 2 + 3 * 3`, and `QN(0) => 2 * 3` (determined from all of the combinations of combining the sectors of the different indices, where the QNs are added and the block dimensions are multiplied). The ordering of the sectors is determined internally by ITensors.jl.

You can also use a combiner on a single Index, which can be helpful for changing the direction of an Index or combining multiple sectors of the same symmetry into a single sector:
```@repl
using ITensors

# This is a short code showing how a combiner
# can be used to "flip" the direction of an Index
i = Index([QN(0) => 2, QN(1) => 3], "i")
j = dag(Index([QN(0) => 2, QN(1) => 3], "j"))
A = randomITensor(i, j)
C = combiner(j; tags = "jflip", dir = -dir(j))
inds(A)
inds(A * C)
```
Unless you are writing very specialized custom code with symmetric tensors, this is generally not needed.

