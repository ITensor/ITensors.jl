#include "itensor/all.h"
using namespace itensor;

int main()
    {
    auto s1 = Index(2,"s1,Site");
    auto s2 = Index(2,"s2,Site");
    auto h1 = Index(10,"h1,Link,H");
    auto h2 = Index(10,"h2,Link,H");
    auto h3 = Index(10,"h3,Link,H");
    auto a1 = Index(100,"a1,Link");
    auto a3 = Index(100,"a3,Link");

    auto Ntrial = 100;

    auto L = randomITensor(h1,prime(a1),a1);
    auto R = randomITensor(h3,prime(a3),a3);
    auto H1 = randomITensor(h1,prime(s1),s1,h2);
    auto H2 = randomITensor(h2,prime(s2),s2,h3);
    auto phi = randomITensor(a1,s1,s2,a3);

    TIMER_START(5);
    for(int n = 1; n <= Ntrial; ++n)
        {
        auto phip = L*phi;
        phip *= H1;
        phip *= H2;
        phip *= R;
        }
    TIMER_STOP(5);

    return 0;
    }
