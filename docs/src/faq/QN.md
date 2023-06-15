# Quantum Number Frequently Asked Questions

## Can I mix different types of quantum numbers within the same system?

Yes, you can freely mix quantum numbers (QNs) of different types. For example,
you can make the sites of your systems alternate between sites carrying
spin "Sz" QNs and fermion sites carrying particle number "Nf" QNs. The QNs will
not mix with each other and will separately be conserved to the original value
you set for your initial wavefunction.

## How can I separately conserve QNs which have the same name?

If you have two physically distinct types of sites, such as "Qudit" sites, but
which carry identically named QNs called "Number", and you want the qudit number
to be separately conserved within each type of site, 
you must make the QN names different for the two types of sites.

For example, the following line of code will make an array of site indices with the qudit number QN having the name "Number\_odd" on odd sites and "Number\_even" on even sites:
```
sites = [isodd(n) ? siteind("Qudit", n; dim=10, conserve_qns=true, qnname_number="Number_odd") 
                  : siteind("Qudit", n; dim=2, conserve_qns=true, qnname_number="Number_even") 
                  for n in 1:2*L]
```
(You may have to collapse the above code into a single line for it to run properly.)
