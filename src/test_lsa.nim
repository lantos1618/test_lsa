
import sequtils, sets, math, strutils, sugar, random



proc calculateTFIDF(docs: seq[seq[string]]): seq[seq[float]] =
  var vocab: HashSet[string]
  for doc in docs:
    var doc_vocab = doc.toHashSet() 
    vocab = vocab + doc_vocab

  var vocab_ordered = vocab.toSeq().toOrderedSet()

  var term_doc_matrix = newSeqWith(vocab_ordered.len, newSeqWith(docs.len, 0))
  for i, doc in docs.pairs:
    for term in doc:
      term_doc_matrix[vocab_ordered.find(term)][i] += 1

  var doc_freq = newSeqWith(term_doc_matrix.len, 0)
  for i in 0..term_doc_matrix.high:
    for j in 0..term_doc_matrix[i].high:
      if term_doc_matrix[i][j] > 0:
        doc_freq[i] += 1

  var idf = newSeqWith(doc_freq.len, 0.0)
  for i in 0..doc_freq.high:
    idf[i] = math.log10(float(docs.len) / float(doc_freq[i]))

  var tf_idf = newSeqWith(term_doc_matrix.len, newSeqWith(term_doc_matrix[0].len, 0.0))
  for i in 0..tf_idf.high:
    for j in 0..tf_idf[i].high:
      tf_idf[i][j] = float(term_doc_matrix[i][j]) * idf[i]

  return tf_idf

# Helper function 
# Yes we could be using a linear algebra, blas, cuda accelerated
proc transpose(A: seq[seq[float]]): seq[seq[float]] =
  result = newSeqWith(A[0].len, newSeq[float](A.len))
  for i, row in A:
    for j, value in row:
      result[j][i] = value
  return result

block: 
  # Define a test matrix
  let A = @[ @[1.0, 2.0, 3.0], @[4.0, 5.0, 6.0], @[7.0, 8.0, 9.0] ]
  
  # Compute the transpose
  let A_T = transpose(A)
  
  # Print the original and transposed matrices
  echo "Original matrix:"
  for row in A:
    echo row
  
  echo "\nTransposed matrix:"
  for row in A_T:
    echo row
  
# yes we could have a matmul
proc matmul(A: seq[seq[float]], B: seq[seq[float]]): seq[seq[float]] =
  # Initialize the result matrix with zeros
  result = newSeqWith(A.len, newSeqWith(B[0].len, 0.0))

  # Perform the matrix multiplication
  for i in 0 ..< A.len:
    for j in 0 ..< B[0].len:
      for k in 0 ..< B.len:
        result[i][j] += A[i][k] * B[k][j]

  return result


block:
  # Define two test matrices
  let A = @[ @[1.0, 2.0, 3.0], @[4.0, 5.0, 6.0] ]
  let B = @[ @[7.0, 8.0], @[9.0, 10.0], @[11.0, 12.0] ]
  
  # Compute the matrix product
  let C = matmul(A, B)
  
  # Print the result
  echo "A * B = "
  for row in C:
    echo row
  

# fancy words for just scaling the range to 1 unit
proc normalize(v: seq[float]): seq[float] =
  let norm = sqrt(v.mapIt(pow(it, 2)).foldl(a + b))
  return v.mapIt(it / norm)


block:
  # Test the normalize function
  let v = @[3.0, 4.0]
  let normalized_v = normalize(v)
  echo "Normalized vector: \n", normalized_v



# Define the dotProduct function
proc dotProduct(v1: seq[float], v2: seq[float]): float =
  return v1.zip(v2).mapIt(it[0] * it[1]).foldl(a + b)


block:
  # Test the dotProduct function
  let v1 = @[1.0, 2.0, 3.0]
  let v2 = @[4.0, 5.0, 6.0]
  let dot_product = dotProduct(v1, v2)
  echo "Dot product: \n", dot_product


proc matvecmul(A: seq[seq[float]], b: seq[float]): seq[float] =
  result = newSeqWith(A.len, 0.0)
  for i in 0 .. A.high:
    result[i] = dotProduct(A[i], b)
  return result


proc powerIteration(A: seq[seq[float]], numIterations: int): (float, seq[float]) =
  randomize()  # Initialize the random number generator
  var b = newSeqWith(A.len, rand(1.0))  # Start with a random vector

  for i in 1 .. numIterations:
    let bPrime = matvecmul(A, b)  # Multiply A by b
    b = normalize(bPrime)  # Normalize bPrime to get the new b

  let eigenvalue = dotProduct(b, matvecmul(A, b))  # The eigenvalue is the dot product of b and Ab
  return (eigenvalue, b)




block:
  # Define a test matrix
  let A = @[ @[1.0, 2.0, 3.0], @[4.0, 5.0, 6.0], @[7.0, 8.0, 9.0] ]
  
  # Compute the largest eigenvalue and corresponding eigenvector
  let (eigenvalue, eigenvector) = powerIteration(A, 20)
  
  # Print the result
  echo "Largest eigenvalue: ", eigenvalue
  echo "Corresponding eigenvector: ", eigenvector

proc svd(A: seq[seq[float]], numIterations: int): (seq[seq[float]], seq[seq[float]], seq[seq[float]]) =
  let A_transpose = transpose(A)
  let AtA = matmul(A_transpose, A)
  let AAt = matmul(A, A_transpose)

  # Eigenvalue and eigenvector for AtA
  var (eigenvalue_1, eigenvector_1) = powerIteration(AtA, numIterations)

  # Eigenvalue and eigenvector for AAt
  var (eigenvalue_2, eigenvector_2) = powerIteration(AAt, numIterations)

  # Singular values are square roots of the eigenvalues
  let singular_values = @[sqrt(eigenvalue_1), sqrt(eigenvalue_2)]

  # U and V are formed from the eigenvectors
  let U = @[eigenvector_2]
  let V = @[eigenvector_1]

  # Sigma is a diagonal matrix formed from the singular values
  var Sigma = newSeqWith(len(A), newSeqWith(len(A[0]), 0.0))
  for i in 0 ..< len(singular_values):
    Sigma[i][i] = singular_values[i]

  return (U, Sigma, V)


block: 
  var docs = @[
      "The cat sat on thek mat.",
      "The dog sat on the log.",
      "Cats and dogs are mortal enemies.",
      "You can teach an old dog new tricks.",
      "The quick brown fox jumps over the lazy dog."
  ]

  # Preprocess the documents
  var docs1 = docs.mapIt(it.filter(x => x.isAlphaAscii() or x == ' ').toLower().join().splitWhitespace())
  
  # Calculate TF-IDF
  var tf_idf = calculateTFIDF(docs1)

  let numIterations = 1000  # Number of iterations for power iteration method

  let (U, Sigma, V) = svd(tf_idf, numIterations)  # Perform SVD on the TF-IDF matrix
  
  echo "U: ", U
  echo "Sigma: ", Sigma
  echo "V: ", V
    
# Function to calculate cosine similarity
proc cosineSimilarity(vec1, vec2: seq[float]): float =
  let dotProd = dotProduct(vec1, vec2)
  let normProd = sqrt(dotProduct(vec1, vec1) * dotProduct(vec2, vec2))
  return dotProd / normProd



var docs = @[
    "The cat sat on thek mat.",
    "The dog sat on the log.",
    "Cats and dogs are mortal enemies.",
    "You can teach an old dog new tricks.",
    "The quick brown fox jumps over the lazy dog."
]

# Calculate TF-IDF for a single document given the vocabulary and idf values
proc calculateSingleDocTFIDF(doc: seq[string], vocab_ordered: seq[string], idf: seq[float]): seq[float] =
  var term_freq = newSeqWith(vocab_ordered.len, 0.0)
  for term in doc:
    let idx = vocab_ordered.find(term)
    if idx != -1:  # Check if term exists in vocabulary
      term_freq[idx] += 1.0

  # Now compute TF-IDF for the document
  for i in 0 ..< term_freq.len:
    term_freq[i] *= idf[i]

  return term_freq

block:
  # Preprocess the documents
  var docs1 = docs.mapIt(it.filter(x => x.isAlphaAscii() or x == ' ').toLower().join().splitWhitespace())

  # Calculate TF-IDF for original documents
  var tf_idf_original = calculateTFIDF(docs1)

  # Calculate the vocabulary and idf for reuse
  var vocab: HashSet[string]
  for doc in docs1:
    var doc_vocab = doc.toHashSet() 
    vocab = vocab + doc_vocab

  var vocab_ordered = vocab.toSeq().toOrderedSet()

  var doc_freq = newSeqWith(tf_idf_original.len, 0)
  for i in 0..tf_idf_original.high:
    for j in 0..tf_idf_original[i].high:
      if tf_idf_original[i][j] > 0:
        doc_freq[i] += 1

  var idf = newSeqWith(doc_freq.len, 0.0)
  for i in 0..doc_freq.high:
    idf[i] = math.log10(float(docs.len) / float(doc_freq[i]))

  # Let's introduce a new document
  var new_doc = "Cats are like dogs."
  var new_doc_preprocessed = new_doc.filter(x => x.isAlphaAscii() or x == ' ').toLower().join().splitWhitespace()
  var tf_idf_new_doc = calculateSingleDocTFIDF(new_doc_preprocessed, vocab_ordered.toSeq, idf)

  # Calculate cosine similarity
  for i, doc in docs1.pairs:
    let similarity = cosineSimilarity(tf_idf_original[i], tf_idf_new_doc)
    echo "Similarity between document ", i+1, " and the new document: ", similarity
