# Affine Cipher

## Problem Statement

Implement the Affine Cipher, an ancient encryption system from the Middle East. The cipher is a type of monoalphabetic substitution cipher where each character is mapped to its numeric equivalent, encrypted with a mathematical function, and then converted back to a letter.

## Implementation Requirements

### Functions to Implement

1. `encrypt(plaintext, a, b)`
   - Encrypts the input string using the affine cipher
   - Returns the encrypted string in lowercase, grouped into 5-letter chunks separated by spaces

2. `decrypt(ciphertext, a, b)`
   - Decrypts the input string using the affine cipher
   - Returns the decrypted string in lowercase, with no spaces

### Mathematical Operations

1. **Encryption Formula**: `E(x) = (a * x + b) mod 26`
   - Where `x` is the letter's index (0-25 for a-z)

2. **Decryption Formula**: `D(y) = a^-1 * (y - b) mod 26`
   - Where `a^-1` is the modular multiplicative inverse of `a mod 26`
   - You must implement the modular multiplicative inverse function

### Input/Output Handling

1. **Input Processing**:
   - Convert all input to lowercase
   - Ignore non-alphabetic characters
   - For decryption, remove all spaces before processing

2. **Output Format**:
   - Encryption: lowercase letters in groups of 5, separated by spaces
   - Decryption: lowercase letters with no spaces

### Error Handling

- If `a` is not coprime with 26, throw an error with the message: "a and m must be coprime."
- Handle edge cases (empty strings, invalid inputs) gracefully

## Examples

### Encryption Examples
```javascript
encrypt("test", 5, 7)       // Returns "ybty"
encrypt("Hello, World!", 5, 7)  // Returns "dahhk hsk"
encrypt("test", 6, 7)       // Throws Error: "a and m must be coprime."
```

### Decryption Examples
```javascript
decrypt("ybty", 5, 7)       // Returns "test"
decrypt("dahhk hsk", 5, 7)  // Returns "helloworld"
decrypt("test", 13, 5)      // Throws Error: "a and m must be coprime."
```

## Constraints
- 0 < a < 26
- 0 ≤ b < 26
- Input text will only contain printable ASCII characters
- Output will be in lowercase

## Implementation Notes

1. **Modular Multiplicative Inverse**:
   - Implement a function to find the modular multiplicative inverse of `a mod 26`
   - The MMI of `a` is the number `x` such that `(a * x) mod 26 = 1`

2. **Efficiency**:
   - Your solution should efficiently handle typical inputs
   - Consider precomputing values where possible

3. **Code Quality**:
   - Write clean, well-documented code
   - Include comments for complex logic
   - Use meaningful variable names

## Test Cases

Test cases are provided in separate JSON files for encryption and decryption. Your implementation should pass all test cases to be considered correct.

## Submission

Submit your implementation with both functions properly exported. The test runner will import and test your functions using the provided test cases.

Output your answer to ../swarm-results/SWARMID#/MMDDYY:HHMMMSS-affine-cipher-answer.js

