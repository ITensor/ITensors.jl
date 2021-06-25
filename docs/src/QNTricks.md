# Symmetric (QN Conserving) Tensors: Background and Usage

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

## Block Sparsity and Quantum Numbers

In general, not all blocks that are allowed according to the flux will actually exist in the tensor (which helps in many cases for efficiency). Usually this would happen when the tensor is first constructed and not all blocks are explicitly set:
```@repl
using ITensors

i = Index([QN(0) => 1, QN(1) => 1])
A = ITensor(i', dag(i));
A[2, 2] = 1.0;
@show A;
D, U = eigen(A; ishermitian=true);
@show D;
@show U;
```
If we had set `A[1, 1] = 0.0` as well, then all of the allowed blocks (according to the flux `QN(0)` would exist and would be included in the eigendecomposition:
```@repl
using ITensors

i = Index([QN(0) => 1, QN(1) => 1])
A = ITensor(i', dag(i));
A[2, 2] = 1.0;
A[1, 1] = 0.0;
@show A;
D, U = eigen(A; ishermitian=true);
@show D;
@show U;
```
"Missing" blocks can also occur with tensor contractions, since the final blocks of the output tensor are made from combinations of contractions of blocks from the input tensors, and there is no guarantee that all flux-consistent blocks will end up in the result:
```@repl
using ITensors

i = Index([QN(0) => 1, QN(1) => 1])
j = Index([QN(0) => 1])
A = ITensor(i, dag(j));
A[2, 1] = 1.0;
@show A;
A2 = prime(A, i) * dag(A);
@show A2;
D, U = eigen(A2; ishermitian=true);
@show D;
@show U;
```

