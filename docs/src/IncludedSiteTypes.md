# SiteTypes Included with ITensor

## "S=1/2" SiteType

Site indices with the "S=1/2" site type represent ``S=1/2`` spins with the states
``|\!\uparrow\rangle``, ``|\!\downarrow\rangle``.

Making a single "S=1/2" site or collection of N "S=1/2" sites
```
s = siteind("S=1/2")
sites = siteinds("S=1/2",N)
```

Available keyword arguments for enabling and customizing quantum numbers (QN) subspaces:
- `conserve_qns` (default: false): conserve total ``S^z``
- `conserve_sz` (default: conserve_qns): conserve total ``S^z``
- `conserve_szparity` (default: false): conserve total ``S^z`` modulo two
- `qnname_sz` (default: "Sz"): name of total ``S^z`` QN
- `qnname_szparity` (default: "SzParity"): name of total ``S^z`` modulo two QN
For example:
```
sites = siteinds("S=1/2",N; conserve_szparity=true, qnname_szparity="SzP")
```

Operators associated with "S=1/2" sites can be made using the `op` function,
for example
```
Sz = op("Sz",s)
Sz4 = op("Sz",sites[4])
```

Available operators are exactly the same as those for the "Qubit" site type. Please
see the list of "Qubit" operators below.

## "Qubit" SiteType

Site indices with the "Qubit" site type represent qubits with the states
``|0\rangle``, ``|1\rangle``.

Making a single "Qubit" site or collection of N "Qubit" sites
```
s = siteind("Qubit")
sites = siteinds("Qubit",N)
```

Available keyword arguments for enabling and customizing quantum numbers (QN) subspaces:
- `conserve_qns` (default: false): conserve total qubit parity
- `conserve_parity` (default: conserve_qns): conserve total qubit parity
- `conserve_number` (default: false): conserve total qubit number
- `qnname_parity` (default: "Parity"): name of total qubit parity QN
- `qnname_number` (default: "Number"): name of total qubit number QN
For example:
```
sites = siteinds("Qubit",N; conserve_parity=true)
```

#### "Qubit" and "S=1/2" States

The available state names for "Qubit" sites are:
- `"0"` (aliases: `"Z+"`, `"Zp"`, `"Up"`, `"↑"`) Qubit in the 0 state
- `"1"` (aliases: `"Z-"`, `"Zm"`, `"Dn"`, `"↓"`) Qubit in the 1 state
- `"+"` (aliases: `"X+"`, `"Xp"`) Qubit in the $|+\rangle$ state (+1 eigenvector of $\sigma_x$)
- `"+"` (aliases: `"X-"`, `"Xm"`) Qubit in the $|-\rangle$ state (-1 eigenvector of $\sigma_x$)
- `"i"` (aliases: `"Y+"`, `"Yp"`) Qubit in the $|i\rangle$ state (+1 eigenvector of $\sigma_y$)
- `"-i"` (aliases: `"Y-"`, `"Ym"`) Qubit in the $|-i\rangle$ state (+1 eigenvector of $\sigma_y$)

#### "Qubit" and "S=1/2" Operators

Operators or gates associated with "Qubit" sites can be made using the `op` function,
for example
```
H = op("H",s)
H3 = op("H",sites[3])
```

Single-qubit operators:
- `"X"` (aliases: `"σx"`, `"σ1"`) Pauli X operator
- `"Y"` (aliases: `"σy"`, `"σ2"`) Pauli Y operator
- `"iY"` (aliases: `"iσy"`, `"iσ2"`) Pauli Y operator times i
- `"Z"` (aliases: `"σz"`, `"σ3"`) Pauli Z operator
- `"√NOT"` (aliases: `"X"`)
- `"H"` Hadamard gate
- `"Phase"` (takes optional argument: ϕ=π/2) (aliases: `"P"`, `"S"`)
- `"π/8"` (aliases: `"T"`)
- `"Rx"` (takes argument: θ) Rotation around x axis
- `"Ry"` (takes argument: θ) Rotation around y axis
- `"Rz"` (takes argument: θ) Rotation around z axis
- `"Rn"` (takes arguments: θ, ϕ, λ) (aliases: `"Rn̂"`) Rotation about axis n=(θ, ϕ, λ)
- `"Proj0"` (aliases: `"ProjUp"`, `"projUp"`) Operator $|0\rangle\langle 0|$
- `"Proj1"` (aliases: `"ProjDn"`, `"projDn"`) Operator $|1\rangle\langle 1|$

Spin operators:
- `"Sz"` (aliases: `"Sᶻ"`) Spin z operator $S^z = \frac{1}{2} \sigma_z$
- `"S+"` (alises: `"S⁺"`, `"Splus"`) Raising operator $S^+ = S^x + iS^y$
- `"S-"` (aliases: `"S⁻"`, `"Sminus"`) Lowering operator $S^- = S^x - iS^y$
- `"Sx"` (alises: `"Sˣ"`) Spin x operator $S^x = \frac{1}{2} \sigma_x$
- `"iSy"` (aliases: `"iSʸ"`) i times spin y operator $iS^y = \frac{i}{2} \sigma_y$
- `"Sy"` (aliases: `"Sʸ"`) Spin y operator $S^y = \frac{1}{2} \sigma_y$
- `"S2"` (aliases: "S²"`) Square of spin vector operator $S^2=\vec{S}\cdot\vec{S}=\frac{3}{4} I$
- `"ProjUp"` (aliases: `"projUp"`, `"Proj0"`) Operator $|\!↑\rangle\langle ↑\!|$
- `"ProjDn"` (aliases: `"projDn"`, `"Proj1"`) Operator $|\!↓\rangle\langle ↓\!|$

Two-qubit gates:
- `"CNOT"` (aliases: `"CX"`) Controlled NOT gate
- `"CY"` Controlled Y gate
- `"CZ"` Controlled Z gate
- `"CPHASE"` (aliases: `"Cphase"`) Controlled Phase gate
- `"CRx"` (aliases: `"CRX"`) (takes arguments: θ)
- `"CRy"` (aliases: `"CRY"`) (takes arguments: θ)
- `"CRz"` (aliases: `"CRZ"`) (takes arguments: θ)
- `"CRn"` (aliases: `"CRn̂"`) (takes arguments: θ, ϕ, λ)
- `"SWAP"` (aliases: `"Swap"`) 
- `"√SWAP"` (aliases: `"√Swap"`) 
- `"iSWAP"` (aliases: `"iSwap"`) 
- `"√iSWAP"` (aliases: `"√iSwap"`) 
- `"Rxx"` (aliases: `"RXX"`) (takes arguments: ϕ) Ising (XX) coupling gate
- `"Ryy"` (aliases: `"RYY"`) (takes arguments: ϕ) Ising (YY) coupling gate
- `"Rzz"` (aliases: `"RZZ"`) (takes arguments: ϕ) Ising (ZZ) coupling gate

Three-qubit gates:
- `"Toffoli"` (aliases `"CCNOT"`, `"CCX"`, `"TOFF"`)
- `"Fredkin"` (aliases `"CSWAP"`, `"CSwap"`, `"CS"`)

Four-qubit gates:
- `"CCCNOT"`

## "S=1" SiteType

Site indices with the "S=1" site type represent ``S=1`` spins with the states
``|\!\uparrow\rangle``, ``|0\rangle``, ``|\!\downarrow\rangle``.

Making a single "S=1" site or collection of N "S=1" sites
```
s = siteind("S=1")
sites = siteinds("S=1",N)
```

Available keyword arguments for enabling and customizing quantum numbers (QN) subspaces:
- `conserve_qns` (default: false): conserve total ``S^z``
- `conserve_sz` (default: conserve_qns): conserve total ``S^z``
- `qnname_sz` (default: "Sz"): name of total ``S^z`` QN
For example:
```
sites = siteinds("S=1",N; conserve_sz=true, qnname_sz="TotalSz")
```

#### "S=1" States

The available state names for "S=1" sites are:
- `"Up"` (aliases: `"Z+"`, `"↑"`) spin in the up state
- `"Z0"` (aliases: `"0"`) spin in the Sz=0 state
- `"Dn"` (aliases: `"Z-"`, `"↓"`) spin in the Sz=0 state

#### "S=1" Operators

Operators associated with "S=1" sites can be made using the `op` function,
for example
```
Sz = op("Sz",s)
Sz4 = op("Sz",sites[4])
```

Spin operators:
- `"Sz"` (aliases: `"Sᶻ"`)
- `"Sz2"` Square of `S^z` operator
- `"S+"` (alises: `"S⁺"`, `"Splus"`)
- `"S-"` (aliases: `"S⁻"`, `"Sminus"`)
- `"Sx"` (alises: `"Sˣ"`)
- `"Sx2"` Square of `S^x` operator
- `"iSy"` (aliases: `"iSʸ"`)
- `"Sy"` (aliases: `"Sʸ"`)
- `"Sy2"` Square of `S^y` operator
- `"S2"` (aliases: "S²"`)

## "Boson" SiteType

The "Boson" site type is an alias for the "Qudit" site type. Please
see more information about "Qudit" below:

## "Qudit" SiteType

Making a single "Qudit" site or collection of N "Qudit" sites
```
s = siteind("Qudit")
sites = siteinds("Qudit",N)
```

Available keyword arguments for enabling and customizing quantum numbers (QN) subspaces:
- `dim` (default: 2): dimension of the index (number of qudit or boson values)
- `conserve_qns` (default: false): conserve total qudit or boson number
- `conserve_number` (default: conserve_qns): conserve total qudit or boson number
- `qnname_number` (default: "Number"): name of total qudit or boson number QN
For example:
```
sites = siteinds("Qudit",N; conserve_number=true)
```

#### "Qudit" and "Boson" Operators

Operators associated with "Qudit" sites can be made using the `op` function,
for example
```
A = op("A",s)
A4 = op("A",sites[4])
```

Single-qudit operators:
- `"A"` (aliases: `"a"`)
- `"Adag"` (aliases: `"adag"`, `"a†"`)
- `"N"` (aliases: `"n"`)

Two-qudit operators:
- `"ab"`
- `"a†b"`
- `"ab†"`
- `"a†b†"`

## "Fermion" SiteType

Site indices with the "Fermion" SiteType represent 
spinless fermion sites with the states
``|0\rangle``, ``|1\rangle``, corresponding to zero fermions or one fermion.

Making a single "Fermion" site or collection of N "Fermion" sites
```
s = siteind("Fermion")
sites = siteinds("Fermion",N)
```

Available keyword arguments for enabling and customizing quantum numbers (QN) subspaces:
- `conserve_qns` (default: false): conserve total number of fermions
- `conserve_nf` (default: conserve_qns): conserve total number of fermions
- `conserve_nfparity` (default: conserve_qns): conserve total fermion number parity
- `qnname_nf` (default: "Nf"): name of total fermion number QN
- `qnname_nfparity` (default: "NfParity"): name of total fermion number parity QN
For example:
```
sites = siteinds("Fermion",N; conserve_nfparity=true)
```

#### "Fermion" States

The available state names for "Fermion" sites are:
- `"0"` (aliases: `"Emp"`) unoccupied fermion site
- `"1"` (aliases: `"Occ"`) occupied fermion site

#### "Fermion" Operators

Operators associated with "Fermion" sites can be made using the `op` function,
for example
```
C = op("C",s)
C4 = op("C",sites[4])
```

Single-fermion operators:
- `"N"` (aliases: `"n"`) Density operator
- `"C"` (aliases: `"c"`) Fermion annihilation operator
- `"Cdag"` (aliases: `"cdag"`, `"c†"`) Fermion creation operator
- `"F"` Jordan-Wigner string operator

## "Electron" SiteType

The states of site indices with the "Electron" SiteType correspond to
``|0\rangle``, ``|\!\uparrow\rangle``, ``|\!\downarrow\rangle``, ``|\!\uparrow\downarrow\rangle``.

Making a single "Electron" site or collection of N "Electron" sites
```
s = siteind("Electron")
sites = siteinds("Electron",N)
```

Available keyword arguments for enabling and customizing quantum numbers (QN) subspaces:
- `conserve_qns` (default: false): conserve total number of electrons
- `conserve_sz` (default: conserve_qns): conserve total ``S^z``
- `conserve_nf` (default: conserve_qns): conserve total number of electrons
- `conserve_nfparity` (default: conserve_qns): conserve total electron number parity
- `qnname_sz` (default: "Sz"): name of total ``S^z`` QN
- `qnname_nf` (default: "Nf"): name of total electron number QN
- `qnname_nfparity` (default: "NfParity"): name of total electron number parity QN
For example:
```
sites = siteinds("Electron",N; conserve_nfparity=true)
```

#### "Electron" States

The available state names for "Electron" sites are:
- `"Emp"` (aliases: `"0"`) unoccupied electron site
- `"Up"` (aliases: `"↑"`) electron site occupied with one up electron
- `"Dn"` (aliases: `"↓"`) electron site occupied with one down electron
- `"UpDn"` (aliases: `"↑↓"`) electron site occupied with two electrons (one up, one down)

#### "Electron" Operators

Operators associated with "Electron" sites can be made using the `op` function,
for example
```
Cup = op("Cup",s)
Cup4 = op("Cup",sites[4])
```

Single-fermion operators:
- `"Ntot"` (aliases: `"ntot"`) Total density operator
- `"Nup"` (aliases: `"n↑"`) Up density operator
- `"Ndn"` (aliases: `"n↓"`) Down density operator
- `"Cup"` (aliases: `"c↑"`) Up-spin annihilation operator
- `"Cdn"` (aliases: `"c↓"`) Down-spin annihilation operator
- `"Cdagup"` (aliases: `"c†↑"`) Up-spin creation operator
- `"Cdagdn"` (aliases: `"c†↓"`) Down-spin creation operator
- `"Sz"` (aliases: `"Sᶻ"`) 
- `"Sx"` (aliases: `"Sˣ"`) 
- `"S+"` (aliases: `"Sp"`, `"S⁺"`,`"Splus"`) 
- `"S-"` (aliases: `"Sm"`, `"S⁻"`, `"Sminus"`) 
- `"F"` Jordan-Wigner string operator
- `"Fup"` (aliases: `"F↑"`) Up-spin Jordan-Wigner string operator
- `"Fdn"` (aliases: `"F↓"`) Down-spin Jordan-Wigner string operator

Non-fermionic single particle operators (these do not have Jordan-Wigner string attached,
so will commute within systems such as OpSum or the `apply` function):
- `"Aup"` (aliases: `"a↑"`) Up-spin annihilation operator
- `"Adn"` (aliases: `"a↓"`) Down-spin annihilation operator
- `"Adagup"` (aliases: `"a†↑"`) Up-spin creation operator
- `"Adagdn"` (aliases: `"a†↓"`) Down-spin creation operator


## "tJ" SiteType

"tJ" sites are similar to electron sites, but cannot be doubly occupied
The states of site indices with the "tJ" SiteType correspond to
``|0\rangle``, ``|\!\uparrow\rangle``, ``|\!\downarrow\rangle``.

Making a single "tJ" site or collection of N "tJ" sites
```
s = siteind("tJ")
sites = siteinds("tJ",N)
```

Available keyword arguments for enabling and customizing quantum numbers (QN) subspaces:
- `conserve_qns` (default: false): conserve total number of fermions
- `conserve_nf` (default: conserve_qns): conserve total number of fermions
- `conserve_nfparity` (default: conserve_qns): conserve total fermion number parity
- `qnname_nf` (default: "Nf"): name of total fermion number QN
- `qnname_nfparity` (default: "NfParity"): name of total fermion number parity QN
For example:
```
sites = siteinds("tJ",N; conserve_nfparity=true)
```

#### "tJ" States

The available state names for "tJ" sites are:
- `"Emp"` (aliases: `"0"`) unoccupied site
- `"Up"` (aliases: `"↑"`) site occupied with one up electron
- `"Dn"` (aliases: `"↓"`) site occupied with one down electron

#### "tJ" Operators

Operators associated with "tJ" sites can be made using the `op` function,
for example
```
Cup = op("Cup",s)
Cup4 = op("Cup",sites[4])
```

Single-fermion operators:
- `"Ntot"` (aliases: `"ntot"`) Total density operator
- `"Nup"` (aliases: `"n↑"`) Up density operator
- `"Ndn"` (aliases: `"n↓"`) Down density operator
- `"Cup"` (aliases: `"c↑"`) Up-spin annihilation operator
- `"Cdn"` (aliases: `"c↓"`) Down-spin annihilation operator
- `"Cdagup"` (aliases: `"c†↑"`) Up-spin creation operator
- `"Cdagdn"` (aliases: `"c†↓"`) Down-spin creation operator
- `"Sz"` (aliases: `"Sᶻ"`) 
- `"Sx"` (aliases: `"Sˣ"`) 
- `"S+"` (aliases: `"Sp"`, `"S⁺"`,`"Splus"`) 
- `"S-"` (aliases: `"Sm"`, `"S⁻"`, `"Sminus"`) 
- `"F"` Jordan-Wigner string operator
- `"Fup"` (aliases: `"F↑"`) Up-spin Jordan-Wigner string operator
- `"Fdn"` (aliases: `"F↓"`) Down-spin Jordan-Wigner string operator

Non-fermionic single particle operators (these do not have Jordan-Wigner string attached,
so will commute within systems such as OpSum or the `apply` function):
- `"Aup"` (aliases: `"a↑"`) Up-spin annihilation operator
- `"Adn"` (aliases: `"a↓"`) Down-spin annihilation operator
- `"Adagup"` (aliases: `"a†↑"`) Up-spin creation operator
- `"Adagdn"` (aliases: `"a†↓"`) Down-spin creation operator

