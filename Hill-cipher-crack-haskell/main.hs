import Data.List (permutations)
import Data.List (foldl')
import Data.List (maximumBy)
import Data.Function (on)
import Debug.Trace
import System.Environment


type Hist = [Double]

occurrences :: String -> Char -> Int
occurrences [] _ = 0
occurrences (x:xs) c
  | c == x = 1 + occurrences xs c
  | otherwise = occurrences xs c

normalizeHist :: Hist -> Hist
normalizeHist xs = map (/ total) xs
  where total = sum xs

multHist :: Hist -> Double -> Hist
multHist xs n = [n * x | x <- xs]

divHist :: Hist -> Double -> Hist
divHist xs n = [x / n | x <- xs]

chi2 :: Hist -> Hist -> Double
chi2 exp observed = sum [(((observed !! i) - e) * ((observed !! i) - e))/e | (i, e) <- (zip [0..] exp)]

chi2Eng :: String -> Double
chi2Eng xs = chi2 (multHist engHist (fromIntegral (length xs))) (hist xs)

hist :: String -> Hist
hist str = [fromIntegral (occurrences str c) | c <- ['a'..'z']]

showHist :: Hist -> String
showHist histogram = unlines [ [toEnum (fromEnum 'a' + i)] ++ ": " ++ show count  | (i, count) <- zip [0..] histogram]

strDotVector :: String -> [Int] -> Char
strDotVector xs vec = toEnum (mod (sum [((fromEnum x - fromEnum 'a') * (vec !! i)) | (i, x) <- zip [0..] xs]) 26 + fromEnum 'a')

-- Combines rows of strings to one string
-- ["ace", "bdf"] -> "abcdef"
combineStrRows :: [String] -> String
combineStrRows rows = [(rows !! (i `mod` (length rows))) !! (i `div` (length rows)) | i <- [0..((length rows) * rowLen - 1)]]
  where rowLen = length (rows !! 0)

blockDot :: String -> [Int] -> String
blockDot [] _ = ""
blockDot str block = strDotVector (take (length block) str) block : blockDot (drop (length block) str) block

factorize :: Int -> [Int]
factorize n = factorize' n 2
  where 
    factorize' num divisor
      | divisor >= num = []
      | mod num divisor == 0 = divisor : factorize' num (divisor+1)
      | otherwise = factorize' num (divisor+1)

toBase26Length :: Int -> Int -> [Int]
toBase26Length 0 _ = []
toBase26Length len x = (toBase26Length (len-1) (x `div` 26)) ++ [mod x 26]

forF :: [Int] -> (Int -> IO ())  -> IO ()
forF [] _ = pure ()
forF (x:xs) f = do
  f x
  forF xs f

removeAtIndex :: Int -> [a] -> [a]
removeAtIndex _ []     = []
removeAtIndex idx xs
    | idx < 0          = xs
    | idx >= length xs = xs
    | otherwise        = take idx xs ++ drop (idx + 1) xs

findMinNValues :: Int -> [Double] -> [(Int, Double)]
findMinNValues n arr = (take n (foldr findMinNValuesIdx [] (zip [0..] arr)))
  where
  findMinNValuesIdx :: (Int, Double) -> [(Int, Double)] -> [(Int, Double)]
  findMinNValuesIdx (i, x) xs
    | length xs < n = (i, x) : xs
    | otherwise = if x < (snd (snd maxEl))
      then (i, x) : removeAtIndex (fst maxEl) xs
      else xs
    where 
      maxEl = (maximumBy (compare `on` (\x -> snd $ snd x)) (zip [0..] xs))

engHist = divHist [8.17, 1.49, 2.78, 4.25, 12.70, 2.23, 2.02, 6.09, 7.00, 0.15, 0.77, 4.03, 2.41, 6.75, 7.51, 1.93, 0.10, 5.99, 6.33, 9.06, 2.76, 0.98, 2.36, 0.15, 1.97, 0.07] 100.0

tryFactor :: String -> Int -> IO()
tryFactor str n = do
  putStrLn ("Trying: " ++ (show n))
  let allDecrypted = [blockDot str (toBase26Length n iter) | iter <- [0..(26^n)]]
  let hi2 = [chi2Eng decrypt | decrypt <- allDecrypted]
  let topN = findMinNValues n hi2
  
  putStrLn "Top hi2 values: "
  putStrLn (show topN)
  putStrLn "\n"
  putStrLn (unlines [ (show i) ++ ": " ++ combineStrRows perm | (i, perm) <- (zip [0..] (permutations [allDecrypted !! i | (i,_) <- topN]))])

main = do
  args <- getArgs
  cypher <- getLine
  putStrLn $ "Cypher length: " ++ (show (length cypher))
  let factors = factorize (length cypher)
  putStrLn ("Possible matrix sizes: " ++ show factors)
  tryFactor cypher (factors !! (read (args !! 0) :: Int))
