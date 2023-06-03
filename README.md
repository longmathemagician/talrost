# A mathematics library for embedded scientific computation

_Note: This library is in no way ready for even limited use._

This repository contains Talrost, an experimental library with the goal of providing an ergonomic, expedient, and embedded-ready numerical tower existing at the edges of Rust’s limitations on specialization and type coercion. Will likely require [`libm`](https://crates.io/crates/libm), currently requires `std`.

## Examples of use

### Complex arithmetic
Currently built over `f32` and `f64` scalar types, exposed as `c32` and `c64` accordingly, with limited coercion. Integer types are WIP.
```rust
use talrost::complex::*;

let a: f64 = 0.25;
let mut b: c64 = (0.75, 1.0).into();
b += a;
b /= c64::new([0.5, 0.5]);
assert_eq!(b, "2 + 0i".into());
```

### Polynomials
Supports polynomials in $\mathbb{R}$ (working) and $\mathbb{C}$ (broken).
```rust
use talrost::polynomial::*;
let tol = f32::EPSILON;

// p(x) = 1x^3 + 5x^2 + -14x + 0, has roots -7, 0, 2
let p = Polynomial::<f32, 3>::new([1.0, 5.0, -14.0, 0.0]); 

let y = p.eval(4.0);
assert_eq!(y, 88.0); // p(4) = 88

let r = p.roots(tol);
assert_eq!(r.len(), 3); // count roots
assert_eq!(r, [2.0, 0.0, -7.0]); // verify ordered roots
```

### Vector and Matrices
Vectors are currently supported in $\mathbb{R^n}$, with explicit coercion to row and column matrix types.
```rust
use talrost::{vector::*, matrix::*};

let v1 = Vector::new([1., 2., 3.]);
let v2 = Vector::new([4., 5., 6.]);

assert_eq!(v1.magnitude(), 14_f64.sqrt());
assert_eq!(v1.normalize().magnitude(), 1.);
assert_eq!(v1.row(), Matrix::new([[1., 2., 3.]]));
assert_eq!(v1.column(), Matrix::new([[1.], [2.], [3.]]));

assert_eq!(v1 + v2, Vector::new([5., 7., 9.]));
assert_eq!(v1 - v2, Vector::new([-3., -3., -3.]));
assert_eq!(v1 * 2., Vector::new([2., 4., 6.]));
assert_eq!(2. * v2, Vector::new([8., 10., 12.]));
```

Currently $M\times N$ matrices over $\mathbb{R}$ have limited support.
```rust
use talrost::matrix::*;

let x = Matrix::<2, 3>::new([[1., 2.], [3., 4.], [5., 6.]]);
assert_eq!((x + Matrix::<2, 3>::ZERO), x);

let y = Matrix::<2, 2>::new([[1., 2.], [3., 4.]]);
assert_eq!(
    (Matrix::<2, 2>::IDENTITY * y).determinant(),
    -2.0
);

let a = Matrix::<4, 4>::new([
    [1., 2., 3., 4.],
    [5., 6., 7., 8.],
    [9., 10., 11., 12.],
    [13., 14., 15., 16.],
]);
let b = Matrix::<4, 4>::new([
    [17., 18., 19., 20.],
    [21., 22., 23., 24.],
    [25., 26., 27., 28.],
    [29., 30., 31., 32.],
]);
let c = Matrix::<4, 4>::new([
    [250., 260., 270., 280.],
    [618., 644., 670., 696.],
    [986., 1028., 1070., 1112.],
    [1354., 1412., 1470., 1528.],
]);
assert_eq!(a * b, c);

```
