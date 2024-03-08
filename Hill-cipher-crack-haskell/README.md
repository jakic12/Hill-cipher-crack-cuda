# Hill-cypher-crack-cuda

Jakob Drusany  
63200005

# Teoretična razlaga

Hillova šifra zakodira tekst z uporabo matričnega množenja vsakega bloka. Če je velikost bloka `k`, je vsaka `k`-ta črka torej odvisna samo od prve vrstice enkripcijske matrike. Izkaže se, da je vsaka `k`-ta črka porazdeljena po enaki porazdelitvi kot vse črke. Ker to razdelitev poznamo, lahko dešifriramo vsako `k`-to črko in dobljeno porazdelitev primirjamo z že vnaprej poznano. Za razdaljo med porazdelitvami uporabimo chi kvadrat. Torej moramo iterirati skozi vse možne dekripcijske vektorje dolžine `k` (vsak element je lahko ena izmed 26 črk) in vsakega ocenit s chi kvadrat. Vzamemo najboljših `k` in poskusimo dekriptirati tekst z matrikami vseh možnih permutacij teh vektorjev. Na koncu moramo dekriptiran tekst ročno preveriti, če je pravilno dekriptirano.

# Razlaga programa

## Uporaba programa

Program je napisan v haskell-u. Program prevedemo z ukazom

```
ghc main.hs
```

Kriptogram prebere iz standardnega vhoda. Zaženemo ga z ukazom

```
cat in.txt | ./main 0
```

ki pove programu naj uporabi 0-ti delitelj dolžine vhoda. V našem primeru na dolžine 604 so možni [2,4,151,302], torej bo program vzel 2. Izpisal nam bo možne čistopise za dan kriptogram, ključ določene dolžine.

Izpis programa:

```
Cypher length: 604
Possible matrix sizes: [2,4,151,302]
Trying: 2
Top hi2 values:
[(1,39.006209116349666),(408,36.18886053422795)]


0: cryptographypriortothemodernagewaseffectivelysynonymouswithencryptiontheconvers...
1: rcpyotrgpayhrpoitrtoehomednrgawesafeeftcvilesynynomyuowstiehcnyrtpoitnehocvnrei...
```

Iz izpisa lahko vidimo, da je 0-ta permutacija izpisala čistopis. Tekstu še dodamo presledke: `cryptography prior to the modern age was effectively synonymous with encryption the convers...`
Če bi hoteli pridobiti dekripcijski ključ, bi morali sestaviti matriko z vrsticami 1 in 408 base 26, permutirati z permutacijo 0 in naredeti inverz te matrike v mod 26.

## Razlaga programa

`n` predstavlja dolžino ključa, `str` pa kriptogram. Najprej dekriptira kriptogram z vsemi možnimi ključi dolžine `n`. To naredi tako, da gre skozi vsa števila od 0 do 26^n, vsakega pretvori v bazo 26 in bločno pomnoži kriptogram z dobljenim vektorjem.

```haskell
let allDecrypted = [blockDot str (toBase26Length n iter) | iter <- [0..(26^n)]]
```

Nato izračuna histogram vsakega niza in izračuna chi2 razdaljo od histograma angleškega jezika.

```haskell
chi2 :: Hist -> Hist -> Double
chi2 exp observed = sum [(((observed !! i) - e) * ((observed !! i) - e))/e | (i, e) <- (zip [0..] exp)]

chi2Eng :: String -> Double
chi2Eng xs = chi2 (multHist engHist (fromIntegral (length xs))) (hist xs)
```

```haskell
let hi2 = [chi2Eng decrypt | decrypt <- allDecrypted]
let topN = findMinNValues n hi2
```

Nakoncu še izpiše vse možne permutacije nizov združene in prebrane po stolpcih

```haskell
putStrLn (unlines [ (show i) ++ ": " ++ combineStrRows perm | (i, perm) <- (zip [0..] (permutations [allDecrypted !! i | (i,_) <- topN]))])
```

# Hitrost programa

```
./main 0  0.33s user 0.02s system 97% cpu 0.361 total
```

Program je potreboval `0.33` sekunde.
