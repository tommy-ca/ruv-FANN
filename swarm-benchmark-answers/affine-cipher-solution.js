// Function to find modular multiplicative inverse
function modInverse(a, m) {
  a = ((a % m) + m) % m;
  for (let x = 1; x < m; x++) {
    if ((a * x) % m === 1) {
      return x;
    }
  }
  throw new Error('a and m must be coprime.');
}

// Function to check if two numbers are coprime
function isCoprime(a, b) {
  while (b !== 0) {
    let temp = b;
    b = a % b;
    a = temp;
  }
  return a === 1;
}

// Encrypt function
export function encrypt(plaintext, a, b) {
  if (!isCoprime(a, 26)) {
    throw new Error('a and m must be coprime.');
  }

  const result = [];
  let count = 0;
  
  for (const char of plaintext.toLowerCase()) {
    if (!/[a-z]/.test(char)) continue;
    
    const x = char.charCodeAt(0) - 97; // 'a' is 0
    const encrypted = (a * x + b) % 26;
    result.push(String.fromCharCode(encrypted + 97));
    
    // Add space every 5 characters for output
    if (++count % 5 === 0) {
      result.push(' ');
    }
  }
  
  return result.join('').trim();
}

// Decrypt function
export function decrypt(ciphertext, a, b) {
  if (!isCoprime(a, 26)) {
    throw new Error('a and m must be coprime.');
  }

  const aInverse = modInverse(a, 26);
  const result = [];
  
  for (const char of ciphertext.toLowerCase().replace(/\s+/g, '')) {
    if (!/[a-z]/.test(char)) continue;
    
    const y = char.charCodeAt(0) - 97; // 'a' is 0
    const decrypted = (aInverse * (y - b + 26)) % 26;
    result.push(String.fromCharCode(decrypted + 97));
  }
  
  return result.join('');
}
