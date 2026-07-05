use talrost::{
    complex::c64,
    dual::{Dual, DualN},
    lattice,
    matrix::Matrix,
    mvpoly::{MPoly, MSystem, Monomial},
    polynomial::Polynomial,
    solvers,
    vector::Vector,
};

// `z / z` at extreme magnitude is the whole point of the Smith-division
// demonstration below; clippy's `eq_op` cannot know that.
#[allow(clippy::eq_op)]
fn complex() {
    let a: f64 = 0.25;
    let mut b: c64 = (0.75, 1.0).into();
    b += a;
    b /= c64::new(0.5, 0.5);
    assert_eq!(b, "2 + 0i".parse().unwrap());

    // Division goes through Smith's algorithm: it stays exact where the
    // textbook (re² + im²) form overflows to garbage.
    let big = c64::new(1e300, 1e300);
    assert_eq!(big / big, c64::new(1.0, 0.0));

    // Polar helpers: binomial start solutions are radius-scaled roots of
    // unity.
    let w = c64::nth_root_of_unity(1, 4); // e^(2πi/4) = i
    assert!((w - c64::new(0.0, 1.0)).magnitude() < 1e-15);
    let z = c64::from_polar(2.0, std::f64::consts::FRAC_PI_2);
    assert!((z - c64::new(0.0, 2.0)).magnitude() < 1e-15);
    assert!((z.arg() - std::f64::consts::FRAC_PI_2).abs() < 1e-15);
}

fn polynomial() {
    let tol = f64::EPSILON;

    // p(x) = x^3 + 5x^2 - 14x, has roots -7, 0, 2.
    // Coefficients are stored ascending: c[i] multiplies x^i.
    let p = Polynomial::new([0.0, -14.0, 5.0, 1.0]);

    let y = p.eval(4.0);
    assert_eq!(y, 88.0); // p(4) = 88

    // Roots contract: ascending order, len() == number of real roots found.
    let r = p.roots(tol);
    assert_eq!(r.len(), 3); // count roots (via Deref to [f64])
    assert_eq!(r.as_slice(), &[-7.0, 0.0, 2.0]); // verify ascending roots
    assert_eq!(r[0], -7.0); // indexing works via Deref

    // Solver modules keep their own native output (NaN for missing roots).
    let r = solvers::yuksel::roots_cubic(&p, tol);
    assert_eq!(r, [-7.0, 0.0, 2.0]);

    // A quadratic with complex roots has no real roots at all.
    let q = Polynomial::new([1.0, -3.0, 5.0]);
    assert!(q.roots(tol).is_empty());

    // eval_with_error: the value plus a running bound on its own rounding
    // error — a solver stopping rule that needs no arbitrary tolerance.
    let (value, bound) = p.eval_with_error(2.0000001);
    assert!(value.abs() > 0.0 && bound < 1e-13);
}

fn vector() {
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
    assert_eq!(v2 / 2., Vector::new([2., 2.5, 3.]));
    assert_eq!(v1[2], 3.); // Index/IndexMut

    // Dot products are hermitian (the second operand is conjugated), so
    // dot(v, v) == |v|² is real even for complex vectors.
    assert_eq!(v1.dot(&v2), 32.);
    let vc = Vector::new([c64::new(1.0, 2.0), c64::new(-3.0, 0.5)]);
    assert_eq!(vc.dot(&vc), c64::new(14.25, 0.0));

    // 3-D vectors have a vector cross product; 2-D keeps the scalar one.
    assert_eq!(v1.cross(&v2), Vector::new([-3., 6., -3.]));
    assert_eq!(Vector::new([1., 2.]).cross(&Vector::new([3., 4.])), -2.);

    // A complex vector's magnitude is an f64, not a complex number.
    let vc = Vector::new([c64::new(1.0, 0.0), c64::new(2.0, 0.0)]);
    let mag: f64 = vc.magnitude();
    assert_eq!(mag, 5_f64.sqrt());
}

fn matrix() {
    // Matrix<T, M, N> is M rows x N cols: this is a 3x2 matrix.
    let x = Matrix::<f32, 3, 2>::new([[1., 2.], [3., 4.], [5., 6.]]);
    assert_eq!((x + Matrix::ZERO), x);

    let y = Matrix::new([[1., 2.], [3., 4.]]);
    assert_eq!((y * Matrix::<_, 2, 2>::IDENTITY).determinant(), -2.0);
    assert_eq!(y.transpose(), Matrix::new([[1., 3.], [2., 4.]]));
    assert_eq!(y[(1, 0)], 3.); // (row, column) indexing

    // Matrix × vector, and the LU-backed one-shot solve.
    let a = Matrix::new([[2., 1.], [1., 3.]]);
    let x = Vector::new([1., -2.]);
    let b = a * x;
    assert_eq!(b, Vector::new([0., -5.]));
    let solved = a.solve(&b).unwrap();
    assert!((solved - x).magnitude() < 1e-12);

    // Factor once, reuse for many right-hand sides (Newton corrector shape);
    // determinant and inverse ride on the same factorization.
    let lu = a.lu().unwrap();
    let x2 = lu.solve(&Vector::new([1., 0.]));
    assert!((a * x2 - Vector::new([1., 0.])).magnitude() < 1e-12);
    assert_eq!(a.determinant(), 5.0);
    let inv = a.inverse().unwrap();
    assert!((inv * b - x).magnitude() < 1e-12);

    // Integer matrices are first-class for the structural (Ring) operations.
    let e = Matrix::<i64, 2, 2>::new([[1, 2], [3, 4]]);
    assert_eq!(e * Matrix::IDENTITY, e);

    let a = Matrix::new([
        [1., 2., 3., 4.],
        [5., 6., 7., 8.],
        [9., 10., 11., 12.],
        [13., 14., 15., 16.],
    ]);
    let b = Matrix::new([
        [17., 18., 19., 20.],
        [21., 22., 23., 24.],
        [25., 26., 27., 28.],
        [29., 30., 31., 32.],
    ]);
    let c = Matrix::new([
        [250., 260., 270., 280.],
        [618., 644., 670., 696.],
        [986., 1028., 1070., 1112.],
        [1354., 1412., 1470., 1528.],
    ]);
    assert_eq!(a * b, c);
}

fn autodiff() {
    // One Horner body for every evaluation domain (`Algebra<T>`):
    // p(x) = x^3 - 2x + 5, ascending storage.
    let p = Polynomial::new([5.0, -2.0, 0.0, 1.0]);

    // Plain evaluation and evaluation at a complex point of the same *real*
    // polynomial.
    assert_eq!(p.eval_at(2.0), p.eval(2.0));
    let z = p.eval_at(c64::new(0.0, 1.0)); // p(i) = -i - 2i + 5 = 5 - 3i
    assert_eq!(z, c64::new(5.0, -3.0));

    // Derivative via a dual number: seed der = 1 and evaluate.
    // p'(x) = 3x^2 - 2, so p'(2) = 10.
    let d = p.eval_at(Dual::variable(2.0));
    assert_eq!(d.val, p.eval(2.0));
    assert_eq!(d.der, 10.0);

    // Vector-mode duals give a full Jacobian in one sweep. The system
    //   f1 = x^2 + y^2 - 5,  f2 = xy - 2      (root at (2, 1))
    // has J = [[2x, 2y], [y, x]].
    let f1 = MPoly::<f64, 2, 3>::new(
        [1.0, 1.0, -5.0],
        [
            Monomial::new([2, 0]),
            Monomial::new([0, 2]),
            Monomial::new([0, 0]),
        ],
    );
    let f2 = MPoly::<f64, 2, 3>::new(
        [1.0, -2.0, 0.0],
        [
            Monomial::new([1, 1]),
            Monomial::new([0, 0]),
            Monomial::new([0, 0]), // rows are padded to a common term count
        ],
    );
    let sys = MSystem::new([f1, f2]);

    let (vals, jac) = sys.eval_jacobian(&[2.0, 1.0]);
    assert_eq!(vals, [0.0, 0.0]);
    assert_eq!(jac, Matrix::new([[4.0, 2.0], [1.0, 2.0]]));

    // The same Jacobian by hand through DualN, for the skeptical.
    let x = DualN::<f64, 2>::variable(2.0, 0);
    let y = DualN::<f64, 2>::variable(1.0, 1);
    let f2_by_hand = x * y - 2.0;
    assert_eq!(f2_by_hand.der, [1.0, 2.0]);

    // A Newton corrector step: solve J·dx = -f at a perturbed point via LU.
    let x0 = [2.1, 0.9];
    let (h, j) = sys.eval_jacobian(&x0);
    let dx = j.solve(&Vector::new([-h[0], -h[1]])).unwrap();
    let x1 = [x0[0] + dx.b[0], x0[1] + dx.b[1]];
    let (r0, r1) = (sys.eval(&x0), sys.eval(&x1));
    assert!(r1[0].abs() + r1[1].abs() < 0.1 * (r0[0].abs() + r0[1].abs()));
}

fn lattice() {
    // Smith normal form of an integer exponent matrix: U·A·V == S with
    // unimodular U, V. |det A| = ∏ Sᵢᵢ = the number of start solutions of
    // the binomial system x^A = b.
    let a = Matrix::<i64, 2, 2>::new([[3, 1], [1, 3]]);
    let (u, s, v) = lattice::smith_normal_form(&a);
    assert_eq!(u * a * v, s);
    assert_eq!(s, Matrix::new([[1, 0], [0, 8]])); // 8 start solutions

    // Hermite normal form: the row-echelon analogue over ℤ (U·A = H).
    let (u, h) = lattice::hermite_normal_form(&a);
    assert_eq!(u * a, h);
}

fn main() {
    complex();
    polynomial();
    vector();
    matrix();
    autodiff();
    lattice();
    println!("demo: all assertions passed");
}
