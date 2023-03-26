# Hill-cypher-crack-cuda
Jakob Drusany
63200005

# Uporaba programa
Program je potrebno prevesti z compiler driverjem `nvcc`.
```
nvcc program.cu -o program
```
Program prebere kriptogram iz standardnega vhoda.
```
echo "kriptogram" | ./program
```

# Teoretična razlaga
Hillova šifra zakodira tekst z uporabo matričnega množenja vsakega bloka. Če je velikost bloka `k`, je vsaka `k`-ta črka torej odvisna samo od prve vrstice enkripcijske matrike. Izkaže se, da je vsaka `k`-ta črka porazdeljena po enaki porazdelitvi kot vse črke. Ker to razdelitev poznamo, lahko dešifriramo vsako `k`-to črko in dobljeno porazdelitev primirjamo z že vnaprej poznano. Za razdaljo med porazdelitvami uporabimo chi kvadrat. Torej moramo iterirati skozi vse možne dekripcijske vektorje dolžine `k` (vsak element je lahko ena izmed 26 črk) in vsakega ocenit s chi kvadrat. Vzamemo najboljših `k` in poskusimo dekriptirati tekst z matrikami vseh možnih permutacij teh vektorjev. Na koncu moramo dekriptiran tekst ročno preveriti, če je pravilno dekriptirano.

# Razlaga programa
Zdelo se mi je, da se da problem precej dobro paralelizirati in sem ga zato implementiral v `C++` za `CUDA` platformo.  
Glavni del programa je ločen v dva koraka:  
## 1. korak: izracunaj vse mozne vektorje dolzine `k` in jih oceni s chi kvadrat
Funkcija, ki se izvaja na grafičnem procesorju se imenuje kernel in jo je potrebno definirati s predpono `__global__`. To funkcijo lahko potem kličemo s posebno sintakso, da grafični procesor zažene veliko število instanc na enkrat. Primer `function<<10,32>>(arg1)`, bo pognal funkcijo `10*32` krat. Vsaka instanca se kliče s svojim indeksom. Ta indeks si lahko predstavljamo kot iterator.  
Torej v prvem koraku moram iterirati skozi vse možne vektorje dolžine `k`. Ker je iterator vedno samo število, se moral iz množice `k^26` števil nekako preslikati v vse možne vektorje dolžine `k`. To sem dosegel, da sem iterator (`i`) pretvoril v bazo 26 in ga shranil v `vec`.

```c
__global__
void get_chi_squared_for_all(
    const char* cypher, // the cypher
    int N, // length of the cypher
    int block_size, // size of the bruteforced vector
    int num_of_vecs, // number of vectors to test
    float* chi_squared // the out chi squared values
){
    // its fine to have this on the device because it is constant
    static const float dist[26] = {8.12f, 1.49f, 2.71f, 4.32f, 12.02f, 2.30f, 2.03f, 5.92f, 7.31f, 0.10f, 0.69f, 3.98f, 2.61f, 6.95f, 7.68f, 1.82f, 0.11f, 6.02f, 6.28f, 9.10f, 2.88f, 1.11f, 2.09f, 0.17f, 2.11f, 0.07f};

    int i = blockIdx.x*blockDim.x+threadIdx.x;

    // cuda will run more threads than we need
    if (i < num_of_vecs) {
        // get vector that is being tested
        int* vec = new int[block_size];
        to_base_N(26, i, vec, block_size);
...
```
Nato sem iteriral skozi vse bloke teksta, skalarno pomnožil blok z vektorjem mod 26 in rezultate zapisal v seznam pojavitev vsake od dekriptiranih črk.
```c
...
        chi_squared[i] = 0.0f;
        int* freq = new int[26];
        int out_length = 0;

        for(int j = 0; j < N; j += block_size){
            int sum = 0;

            // multiply the vector with the block
            for(int l = 0; l < block_size; l++){
                sum += (cypher[j+l] - 'a') * vec[l];
            }
            sum %= 26;

            // add to the frequency
            freq[sum]++;
            out_length++;
        }
...
```
Na koncu sem še izračunal chi kvadrat med dobljeno porazdelitvijo in pričakovano.
```c
...
        // calculate chi squared
        for(int j = 0; j < 26; j++){
            float expected = dist[j]/100.0f * out_length;
            chi_squared[i] += (freq[j] - expected)*(freq[j] - expected)/expected;
        }

        //printf("[%d] chi squared done\n", i);

        delete[] vec;
        delete[] freq;
    }

}
```
## 2. korak: dekriptiraj program z vsemi možnimi permutacijami vektorjev
Podobno kot prej, je potrebno sedaj iterirati skozi vse možne permutacije. To sem dosegel, da sem iterator pretvoril v faktorielno bazo. Funkcija torej generira permutacijo `k` števil iz določenega indeksa permutacije.
```c
void base_factorial_to_perm(
    int perm_i, // the index of the permutation in the factorial base
    int* out, // the output array
    int size // the size of the arrays
){
    // pool of numbers to chose
    int* temp = new int[size];
    for(int i = 0; i < size; i++){
        temp[i] = i;
    }

    // this operation can be slower, because size is small
    for(int i = 0; i < size; i++){
        // convert from base factiorial to base 10
        // (each digit is the index of the number to chose from the pool)
        int j = perm_i % (size - i);
        perm_i = perm_i / (size - i);

        // chose the number
        out[i] = temp[j];

        // remove the number from the pool
        for(int l = j; l < size - i - 1; l++){
            temp[l] = temp[l+1];
        }
    }
}
```
Želimo dekriptirati kriptogram dolžine `N` z vsemi možnimi permutacijami `k` vektorjev. Torej želimo matriko velikosti `k! * N`. Za paralelizacijo tega procesa, se dekripcijska funkcija požene za vsako permutacijo in za vsak blok v tekstu paralelno. Funkcija ki dekriptira blok `j` za permutacijo vektorjev z indeksom `i`.
```c
__global__
void get_decrypted_permutations(
    const char* cypher, // the cypher
    int N, // length of the cypher
    int block_size, // size of the matrix
    int num_of_perms, // number of permutations
    int* vectors, // the vectors to test permutations of (vector of numbers)
    char* decrypted // the out decrypted cypher (flattened matrix)
){
    int i = blockIdx.x*blockDim.x+threadIdx.x; // permutation id
    int j = blockIdx.y*blockDim.y+threadIdx.y; // block id (block in text)

    if (i < num_of_perms && j*block_size < N) {

        // get how the vectors are permuted in the permutation encoded as i
        // (array of indexes)
        int* idx_permutation = new int[block_size];
        base_factorial_to_perm(i, idx_permutation, block_size);

        // for each letter in the current block to decypher
        int* vector = new int[block_size];
        for(int m = 0; m < block_size; m++){
            // convert from number to vector
            to_base_N(26, vectors[idx_permutation[m]], vector, block_size);

            // multiply block with vector
            int chr = 0;
            for(int l = 0; l < block_size; l++){
                chr += (cypher[j*block_size+l] - 'a') * vector[l];
            }
            chr = chr % 26;
            decrypted[i*N + (j*block_size+m)] = chr + 'a';
        }

        delete[] idx_permutation;
        delete[] vector;
    }
}
```
Ostali deli programa tečejo na procesorju in so branje vhoda, kopiranje podatkov na RAM grafičnega procesorja, izpis.

# Izpis programa
Izhodi so skrajšani za berljivost
## Kriptogram iz eučilnice
```
N:604
block_size:2
number of vectors to check: 676
(676):done calculating chi
smallest block_size:[26 483 ]
26[0, 1, ]:40.068
483[15, 18, ]:33.8264

0: cryptographypriortothemodernagewaseffectivelysyn...
1: rcpyotrgpayhrpoitrtoehomednrgawesafeeftcvilesyny...
block_size:4
number of vectors to check: 456976
(456976):done calculating chi
smallest block_size:[483 8933 26 9152 ]
483[15, 18, 0, 0, ]:23.8969
8933[15, 5, 13, 0, ]:127.038
26[0, 1, 0, 0, ]:40.9216
9152[0, 14, 13, 0, ]:169.851

0: recpoottppaarepcttrrerhueeddggaasfaneeffviivsfyl...
1: ercpoottppaaerpcttrrrehueeddggaafsaneeffivivfsyl...
2: creptootappaprecrttrherudeedaggaasfnfeefivivysfl...
3: prectootappacreprttruerhdeedaggansfafeefvviilsfy...
4: rcepototpaparpectrtrehruededgagasafnefefviivsyfl...
5: ecrpototpapaeprctrtrrheuededgagafasnefefiivvfysl...
6: cerptootappapercrttrhreudeedaggaafsnfeefiivvyfsl...
7: perctootappacerprttrurehdeedagganfsafeefvivilfsy...
8: rpecototpaparceptrtreurhededgagasnfaefefvviislfy...
9: eprcototpapaecrptrtrruehededgagafnsaefefivviflsy...
10: cprettooaapppcrerrtthuerddeeaaggansfffeeivviyls...
11: pcrettooaappcprerrttuherddeeaaggnasfffeevivilys...
12: repcoottppaarecpttrreruheeddggaasfnaeeffvivisfl...
13: erpcoottppaaercpttrrreuheeddggaafsnaeeffivvifsl...
14: crpetotoapapprcertrtheurdedeagagasnffefeivviysl...
15: prcetotoapapcrpertrtuehrdedeagagnsaffefevviilsy...
16: rcpeottopaaprpcetrrtehureddegaagsanfeffevivisyl...
17: ecprottopaapepcrtrrtrhueeddegaagfanseffeiivvfyl...
18: ceprtotoapappecrrtrthruededeagagafnsfefeiivvyfl...
19: pecrtotoapapceprrtrturhededeagagnfasfefeviivlfy...
20: rpceottopaaprcpetrrteuhreddegaagsnafeffevviisly...
21: epcrottopaapecprtrrtruheeddegaagfnaseffeivivfly...
22: cperttooaapppcerrrtthureddeeaagganfsffeeivivylf...
23: pcerttooaappcperrrttuhreddeeaaggnafsffeeviivlyf...
block_size:151
number of vectors to check: -2147483648
Overflow in number of vectors
```

# Performance
Ker imam prenosnik brez grafične kartice, sem moral testirati z google collab, torej je performance slabši.  
Kriptogram iz eucilnice (dolzine 676)
```
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   97.33%  347.94ms         2  173.97ms  1.8745ms  346.07ms  get_chi_squared_for_all(cha...
                    2.67%  9.5363ms         2  4.7682ms  2.8707ms  6.6656ms  get_decrypted_permutations(...
```
Program je testiral različne dolžine dekripcijskega ključa.  
Najprej velikost 2 (vse skupaj <b>`4.7452ms`</b>):  
 * Za izračun chi kvadrat vseh možnih vektorjev dolžine 2 (676), je program potreboval <b>`1.8745ms`</b>  
 * Za dekripcijo kriptograma z vsemi možnimi permutacijami (2), je program potreboval <b>`2.8707ms`</b>  
 
Za velikost 4 (vse skupaj <b>`352.7356`</b>):
 * Za izračun chi kvadrat vseh možnih vektorjev dolžine 4 (456976), je program potreboval <b>`346.07ms`</b>  
 * Za dekripcijo kriptograma z vsemi možnimi permutacijami (24), je program potreboval <b>`6.6656ms`</b>  
 
## Večji primer
Kriptogram dolžine 1450, zašifriran z ključem dolžine 5
```
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   98.99%  4.14039s         2  2.07019s  2.1595ms  4.13823s  get_chi_squared_for_all(cha...
                    1.01%  42.451ms         2  21.225ms  5.6102ms  36.840ms  get_decrypted_permutations(...
```
Za velikost 5 (vse skupaj <b>`~4s`</b>):
 * Za izračun chi kvadrat vseh možnih vektorjev dolžine 4 (11881376), je program potreboval <b>`4.13823s`</b>  
 * Za dekripcijo kriptograma z vsemi možnimi permutacijami (120), je program potreboval <b>`36.840ms`</b>

# Zaključek
Zdi se mi, da je problem precej dobro paralelizabilen in sem veliko pridobil na hitrosti zaradi implementacije na grafičnem procesorju.

 

