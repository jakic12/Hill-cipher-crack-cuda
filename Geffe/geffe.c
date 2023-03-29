#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// polinomi za LFSRje
char* LFSR1 =       "10010";
char* LFSR2 =     "1000001";
char* LFSR3 = "10000000010";

// koda, ki generira output za LFSR(polynomial) z začetnim stanjem initial_state
// dolžina outputa je len_out
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
 
void to_base_N(int N, int num, char* out, int size) {
    int i = 0;
    while(num > 0) {
        if (i >= size) {
            break;
        }
        out[size-i-1] = num % N;
        num = num / N;
        i++;
    }
    for(; i < size; i++) {
        out[size-i-1] = 0;
    }
}
char from_base_N(int N, char* in, int size) {
		int num = 0;
		for(int i = 0; i < size; i++) {
				num += in[i] * pow(N, size-i-1);
		}
		return num;
}


int main(void)
{
	// vhodni podatki
	char* cipher = "01100111011111111001100011111010101011100111101100011111100100111000111010010110111110110110111010000000011000100110111011001110000110010111010010011000101110001000000101011101011101011110111111110110110111001000100000100011101000000011110010110110100011100101100100111011011111011000100010111010110011101001001111100100100011000111011110001001011111010110011101010011010111010010000010000001000100001101010111010011100100011111010111111011100011000011000001111000101110100110101100011111000110010011100010101100011110101001101000101010101001001111111101101110110111010010110010010111100111111110000000010001010011101000001010010101111000111101100000000111111111010100111010010000110011000111111101011010001001110101101010101101101100101101011101000010011111110010001010101000101011111101110000100101110001110111010011101101110011000111000110000001010111101100000011000111100110101100111010000000111010110111101110000001010100110111000010001111000001110110100000101110010101111110001110010101000100100011000010010100001100001111010110101001111111100110010110110010010000001000110000011010101010100011010111100011100011100001011110111110110111010100010100001010101111101111010011011010101111011111110100100010110110111001101100010100001101101000111100111011101011110010001011010001110000000111000011010011000111000101011100101100001011100011011110110001100001000111011101010101011010101011100100110010001111110001011000011011000100011111001100100110110001100110001000100101100101101010010100010011001111111101010001000111110110100011000000001011000000001101111001000000011011000011001001000001000001100010100110100101111001010101100110111011111001100100101101011110011011101100010110111100110100111110000010000101100011010111000000100111110100111011110001101001011000010101111111001111010001000101001110110000000101001111011010010011010100001101110000110011010101001001011100100110100010010011010111100101110001110000110111001011100100011100101111110111100110111111011110111110111101110010111000101001011101011000110011111011001000110001101010110100100011011010000110011111101000010010111010111010110000010011010000011110111100101110010010001100001010110110100101111000011001011101100010100011111001001011010101111111010101010110101000100101011011110000111100000101011100011110010000110010100000010100100011001000000100101010000011110111000100111010011000100000111001001011000011001011010100001010000001101111001001100100011101111001111101000010011010011111100110001111111011100101111000101010111000110010001000011010011101100000001101010000001101101000100111101000001110111001110100101110001100001110111110101110101111011110000010001100111101101011100101011000001101010101010110000111100010101101011110001000010001111100100101110110110010011110110010011001010001101100101011000001011101010001010110000010101111111100000111100111110010111101001010110000000100101000101111110111011110101110101010111100101000111101000101011000101100110000010000011010101000011001101110111110100111011111011001101001011001100000111011101110";
	char* known  = "000101000111000011111001101110001101000100000011110011111000";

	int len_cipher = strlen(cipher);
	int len_known = strlen(known);

	// xor cipher z known, da dobimo z
	char* z = malloc(len_known);
	for(int i = 0; i < len_known; i++) {
		z[i] = (cipher[i]-'0') ^ (known[i] - '0');
	}

	// definiramo LFSRje
	// vrsti red je pomemben, ker je LFSR2 hocemo testirati, ko ze imamo LFSR1 in LFSR3
	char* LFSRs[3];
	LFSRs[0] = LFSR1;
	LFSRs[1] = LFSR3;
	LFSRs[2] = LFSR2;

	int lfsr_len[3];
	lfsr_len[0] = 5;
	lfsr_len[1] = 11;
	lfsr_len[2] = 7;

	// dekripcijki kljuci generirani z LFSRji (outputi)
	char* dec_key[3]; 
	for(int i = 0; i < 3; i++){
		dec_key[i] = malloc(len_known);
	}
	char* tmp = malloc(len_known);

	// tukaj bomo shranili najboljsa zacetna stanja za LFSRje
	char* initial_states[3];
	for(int i = 0; i < 3; i++){
		initial_states[i] = malloc(lfsr_len[i]);
	}


	// glavni del, ki testira vse mozna zacetna stanja za vse lfsrje
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