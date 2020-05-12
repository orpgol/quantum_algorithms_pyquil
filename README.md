# quantum_algorithms_pyquil

# Usage
In order to run these programs you must have Python 3 and PyQuil 2 properly installed. Alternatively, these can be run in the `rigetti/forest` Docker container.

## Bernstein-Vazirani
```
usage: Bernstein-Vazirani.py [-h] a b

positional arguments:
  a           Enter the a bit string for the function ax(xor)b you want to run
              with
  b           Enter the b sibgle bit for the function ax(xor)b you want to run
              with

optional arguments:
  -h, --help  show this help message and exit
```

## Deutsch-Jozsa
```
usage: Deutsch-Jozsa.py [-h] bits algo

positional arguments:
  bits        Enter the number of bits you want to run with
  algo        Enter balanced for balanced or const for constant function

optional arguments:
  -h, --help  show this help message and exit
```

## Simon's
```
usage: Simons.py [-h] string ftype

positional arguments:
  string      Secret string s of length n
  ftype       1 for one-to-one or 2 for two-to-one

optional arguments:
  -h, --help  show this help message and exit
```

Examples:

`python Simons.py 10 1`

`python Simons.py 100 2`

## Grover
```
usage: Grover.py [-h] bit valid

positional arguments:
  bit         Enter the bit string for your function to search for
  valid       Enter valid for valid grover mapping or invalid for no string
              returning 1

optional arguments:
  -h, --help  show this help message and exit
```

