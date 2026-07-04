use talrost::{complex::c64, matrix::Matrix, polynomial::Polynomial, solvers, vector::Vector};

fn complex() {
    let a: f64 = 0.25;
    let mut b: c64 = (0.75, 1.0).into();
    b += a;
    b /= c64::new(0.5, 0.5);
    assert_eq!(b, "2 + 0i".parse().unwrap());
}

fn polynomial() {
    let tol = f64::EPSILON;

    // p(x) = 1x^3 + 5x^2 + -14x + 0, has roots -7, 0, 2
    let p = Polynomial::new([1.0, 5.0, -14.0, 0.0]);

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

fn main() {
    complex();
    polynomial();
    vector();
    matrix();
    println!("demo: all assertions passed");
}
