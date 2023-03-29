# Hill-cypher-crack-cuda

Jakob Drusany  
63200005

# Uporaba programa

Program je potrebno prevesti z `gcc`.

```
gcc geffe.c -o geffe
```

Program prebere ima kriptogram in poznano besedilo že v kodi, torej če želite pognati na drugem vhodu, je to potrebno spremeniti v kodi

```
./geffe
```

# Teoretična razlaga

Geffejev generator psevdonaključnih števil je sestavljen iz treh pomičnih registrov s povratno zanko: LFSR1, LFSR2 in LFSR3. Označimo izhodne bite posameznih registrov z `x1`, `x2` oziroma `x3`. Potem je izhodni bit generatorja enak `k=x1*x2+x2*x3+x3 (mod 2)`.

Naj bodo karakteristični polinomi LFSR-jev za dani Geffejev generator enaki

```
p1(x)=x5+x2+1
p2(x)=x7+x+1
p3(x)=x11+x2+1
```

Recimo, da smo prestregli kriptogram in poznamo začetek kriptograma (recimo vemo, da je html datoteka in se začne z `<!DOCTYPE html><html>`). Če hočemo dekodirati preostanek sporočila, moramo najti začetna stanja vseh LFSR registrov.  
Ker imamo sistem treh LFSR registrov, bi morali pregledati vse možne kombinacije vseh začetnih stanj. To pomeni, če so `m1,m2,m3` stopnje LFSR1,LFSR2 in LFSR3, bi potrebovali poskusiti `2^(m1+m2+m3)` kombinacij.  
Poglejmo tabelo izhodov `k`, pri vseh kombinacijah `x1`,`x2` in `x3`

```
x1 x2 x3 | k
------------
0  0  0  | 0
0  0  1  | 1
0  1  0  | 0
0  1  1  | 0
1  0  0  | 0
1  0  1  | 1
1  1  0  | 1
1  1  1  | 1
```

lahko vidimo, da se stolpca `x1` in `k` ujemata v 3/4 primerov. Isto vidimo za `x3` in `k`. To pomeni, da je izhod GEFFEjevega generatorja precej podoben samo izhodu LFSR1 in LFSR3.  
To dejstvo lahko izrabimo in najdemo za LFSR1 in LFSR3 začetni stanji, ki bodo 3/4 bitov dekodirali pravilno. To lahko naredimo za vsakega posebej. Ko enkrat imamo ti dve začetni stanji, lahko najdemo še začetno stanje za LFSR3. Poskusimo vsa možna začetna stanja, za vsakega izračunamo `k` in pogledamo v koliko bitih se vjema z našim poznanim besedilom. Ko iščemo začetno stanje za LFSR3, potrebujemo popolno vjemanje.  
S tem smo skrajšali iskanje na `2^m1+2^m2+2^m3` možnosti.

# Razlaga programa

Najprej moremo definirati LFSRje

```c
char* LFSR1 =       "10010";
char* LFSR2 =     "1000001";
char* LFSR3 = "10000000010";
```

Tukaj vsaka enica na `i`-tem mestu (iz desne proti levi) predstavlja koeficient pred `x^(i+1)`  
Koda, ki generira izhod iz LFSR definiranega z polinomom `polynomial`.

```c
void LFSR(char* out, char* initial_state, char* polynomial, int poly_len, int len_out)
{
	// kopiramo začetno stanje v izhod
	for(int i = 0; i < poly_len; i++) {
		out[i] = initial_state[i];
	}
	// izračunamo izhod
	for(int j = 0; j < len_out-poly_len; j++) {
		char chr = 0;
		for(int i = j; i < j+poly_len; i++) {
			// xoramo izhod z polinomom
			if(out[i] == '1') {
				chr ^= (polynomial[i-j]-'0');
			}
		}
		out[j+poly_len] = chr + '0';
	}
}
```

Najprej so definirani vhodni podatki. Izračuna se tudi `z`, ki predstavlja iskan dekripcijski ključ.

```c
int main(void)
{
	// vhodni podatki
	char* cipher = "011001110111111110011000111110101010111001111011000111111001...";
	char* known  = "000101000111000011111001101110001101000100000011110011111000";

	int len_cipher = strlen(cipher);
	int len_known = strlen(known);

	// xor cipher z known, da dobimo z
	char* z = malloc(len_known);
	for(int i = 0; i < len_known; i++) {
		z[i] = (cipher[i]-'0') ^ (known[i] - '0');
	}
  ...
```

Nato definiramo vrsti red, po katerem bomo iskali stanja LFSRjev. Zadnjega bomo obravnavali posebej, saj v zadnjem primeru, imamo že ostale in lahko izračunamo izhod `k`

```c
char* LFSRs[3];
LFSRs[0] = LFSR1;
LFSRs[1] = LFSR3;
LFSRs[2] = LFSR2;

int lfsr_len[3];
lfsr_len[0] = 5;
lfsr_len[1] = 11;
lfsr_len[2] = 7;
```

Glavni del se dogaja, ko testiramo vsa možna začetna stanja za LFSRje. Tukaj najprej iteriramo skozi LFSRje in za vsakega naredimo iterator čez vsa možna stanja. Ta shranimo v `initial_state`. Nato poženemo LFSR in generiran izhod shranimo v `tmp`

```c
  ...

  for(int lfsr = 0; lfsr < 3; lfsr++) {
		char* initial_state = malloc(lfsr_len[lfsr]);
		int max_correct = 0;
		for(int i = 0; i < pow(2,lfsr_len[lfsr]); i++) {
			// gremo skozi vsa mozna zacetna stanja
			to_base_N(2, i, initial_state, lfsr_len[lfsr]);
			for(int j = 0; j < lfsr_len[lfsr]; j++) {
				// pretvorimo iz stevila v string
				initial_state[j] += '0';
			}
			// generiramo dekripcijki kljuc
			LFSR(tmp, initial_state, LFSRs[lfsr], lfsr_len[lfsr], len_known);

      ...
```

Za vsak generiran ključ preverimo koliko bitov se vjema z željenim ključem. V primeru, ko obravnavamo zadnji LFSR, torej ko je `lfsr==2`, izračunamo `k` in tega previrjamo z željenim ključem.

```c
      ...
			// preverimo koliko bitov je pravilnih
			int correct = 0;
			for(int j = 0; j < len_known; j++) {
				char decrypted;

				// ce je LFSR2(ki je na zadnjem indeksu), potem upostevamo tudi LFSR1 in LFSR3
				if(lfsr == 2) {
					// k=x1*x3+x3*x2+x2
					decrypted = ((dec_key[0][j]-'0') & (tmp[j]-'0')) ^ ((tmp[j]-'0') & (dec_key[1][j]-'0')) ^ (dec_key[1][j]-'0');
				} else {
					// prevodimo iz char v int
					decrypted = tmp[j]-'0';
				}

				if(decrypted == z[j]) {
					correct += 1;
				}
			}
      ...
```

shranimo si najboljše ovrednoteno začetno stanje. Informativno izpišemo, če smo našli take, ki se vjemajo v vsaj 3/4 primerih ali v vseh primerih, če obravnavamo zadnji LFSR

```c
      ...
			// ce je najvec pravilnih, potem shranimo to zacetno stanje
			if(correct >= max_correct) {
				max_correct = correct;
				memcpy(dec_key[lfsr], tmp, len_known);
				memcpy(initial_states[lfsr], initial_state, lfsr_len[lfsr]);
			}

			if(lfsr == 2) {
				if(correct == len_known){
					printf("[%d] GEFFE: %d%% correct with initial state '%s'\n", lfsr+1, (int)(correct/(float)len_known*100), initial_state);
					break;
				}
			}else if(correct >= (len_known*3)/4) {
				printf("[%d] LFSR: %d%% correct with initial state '%s'\n", lfsr+1, (int)(correct/(float)len_known*100), initial_state);
				break;
			}
		}

		free(initial_state);
	}
	free(tmp);

  ...
```

Na koncu še dekriptiramo kriptogram z izračunanimi začetnimi stanji.

```c
	// dekripcija
	char* decrypted = malloc(len_cipher);

	// generiramo dekripcijke kljuce
	for(int i = 0; i < 3; i++){
		free(dec_key[i]);
		dec_key[i] = malloc(len_cipher);
		LFSR(dec_key[i], initial_states[i], LFSRs[i], lfsr_len[i], len_cipher);
	}

	char temp_char = 0;
	int power = 4;
	printf("\nDecrypted:\n");
	for(int i = 0; i < len_cipher; i++) {
		// LFSR3 je na indeksu 1 in LFSR2 na indeksu 2
		decrypted[i] = ((dec_key[0][i]-'0') & (dec_key[2][i]-'0')) ^ ((dec_key[2][i]-'0') & (dec_key[1][i]-'0')) ^ (dec_key[1][i]-'0');
		decrypted[i] ^= cipher[i]-'0';

		// sproti pretvorimo dvojiško zaporedje dolžine 5 v črko
		temp_char += (decrypted[i] << power);
		if(power == 0) {
			printf("%c", temp_char+'a');
			temp_char = 0;
			power = 5;
		}
		power -= 1;
		decrypted[i] += '0';
	}
}
```

# Izpis programa

```
[1] LFSR: 76% correct with initial state '01110'
[2] LFSR: 78% correct with initial state '11110011010'
[3] GEFFE: 100% correct with initial state '1101001'

Decrypted:
cryptographypriortothemodernagewaseffectivelysynonymouswithencryptiontheconversionofinformationfromareadablestatetoapparentnonsensetheoriginatorofanencryptedmessagealicesharedthedecodingtechniqueneededtorecovertheoriginalinformationonlywithintendedrecipientsbobtherebyprecludingunwantedpersonsevefromdoingthesamethecryptographyliteratureoftenusesaliceaforthesenderbobbfortheintendedrecipientandeveeavesdropperfortheadversarysincethedevelopmentofrotorciphermachinesinworldwariandtheadventofcomputersinworldwariithemethodsusedtocarryoutcryptologyhavebecomeincreasinglycomplexanditsapplicationmorewidespread
```

Program izpiše stanja za LFSRje, ki ustrezajo našim zahtevam in dekriptiran kriptogram. Ročno moramo še dodati presledke.

# Performance

Za testiranje sem uporabil `hyperfine`, ki zažene program večkrat in zračuna povprečje

```
Benchmark 1: ./geffe
  Time (mean ± σ):      10.1 ms ±   0.9 ms    [User: 8.6 ms, System: 0.7 ms]
  Range (min … max):     9.0 ms …  17.2 ms    192 runs
```

Program torej porabi 10 milisekund, da izračuna rezultat.

## Večji primeri

Meritve na višjih stopnjah LFSRjev (vsi tri iste stopnje)

```
m  | čas (milisekunde)
----------------------
7                    9
10                  30
13                 114
15                 220
18                1569
20                5360
```
