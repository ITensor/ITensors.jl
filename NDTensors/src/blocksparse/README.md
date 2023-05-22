BlockSparse datastructure: 

Blocked structure which *currently* stores information about extent wise blocking and then uses that to form memory Blocks. 
All memory blocks are stored in a single data structure called `data`. This means currently all `blocks` have a single storage type if they are not empty. 

Sparsity comes from not storing empty blocks at all. Right now numbers specifically are stored in data.

TODO:
  Blocks should be associated with storage types and not locations in the data vector. 
  Data vector can still store numbers and storage types should store views of the data in this vector.  
  Also we could do something like store vectors of data in the `data` vector or 