/// Cem Yuksel's polynomial root finder, as described by
/// ``High-Performance Polynomial Root Finding for Graphics'' in
/// Proc. ACM Comput. Graph. Interact. Tech. (Proceedings of HPG 2022)
/// Rust port of the methods in https://github.com/cemyuksel/cyCodeBase/blob/master/cyPolynomial.h
///
/// Provided under the following license:
///
/// MIT License
///
/// Copyright (c) 2016, Cem Yuksel <cem@cemyuksel.com>
/// All rights reserved.
///
/// Permission is hereby granted, free of charge, to any person obtaining a copy
/// of this software and associated documentation files (the "Software"), to deal
/// in the Software without restriction, including without limitation the rights
/// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
/// copies of the Software, and to permit persons to whom the Software is
/// furnished to do so, subject to the following conditions:
///
/// The above copyright notice and this permission notice shall be included in all
/// copies or substantial portions of the Software.
///
/// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
/// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
/// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
/// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
/// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
/// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
/// SOFTWARE.
use crate::polynomial::Polynomial;

#[inline]
fn find_closed(
    N: usize,
    p: impl Fn(f64) -> f64,
    dp: impl Fn(f64) -> f64,
    mut x0: f64,
    mut x1: f64,
    y0: f64,
    y1: f64,
    tol: f64,
) -> f64 {
    let ep2 = 2. * tol;
    let mut xr = (x0 + x1) / 2.;
    if x1 - x0 <= ep2 {
        return xr;
    }

    if N <= 3 {
        let xr0 = xr;
        for _ in 0..16 {
            let mut xn = xr - p(xr) / dp(xr);
            xn = xn.clamp(x0, x1);
            if (xr - xn).abs() <= tol {
                return xn;
            }
            xr = xn;
        }
        if !xr.is_finite() {
            xr = xr0
        }
    }

    let mut yr = p(xr);
    let mut xb0 = x0;
    let mut xb1 = x1;

    loop {
        let side = (y0 < 0.) != (yr < 0.);
        if side {
            xb1 = xr;
        } else {
            xb0 = xr;
        }
        let dy = dp(xr);
        let dx = yr / dy;
        let xn = xr - dx;
        if (xn > xb0) && (xn < xb1) {
            let stepsize = (xr - xn).abs();
            xr = xn;
            if stepsize > tol {
                yr = p(xr);
            } else {
                let mut xs;
                if tol == 0. {
                    xs = if side {
                        xb1 - f64::EPSILON
                    } else {
                        xb0 + f64::EPSILON
                    };
                } else {
                    xs = xn - tol * if side { 1. } else { -1. };
                    if xs == xn {
                        xs = if side {
                            xb1 - f64::EPSILON
                        } else {
                            xb0 + f64::EPSILON
                        };
                    }
                }
                let ys = p(xs);
                let s = (y0 < 0.) != (ys < 0.);
                if side != s {
                    return xn;
                };
                xr = xs;
                yr = ys;
            }
        } else {
            xr = (xb0 + xb1) / 2.;
            if xr == xb0 || xr == xb1 || xb1 - xb0 <= ep2 {
                if tol == 0. {
                    let xm = if side { xb0 } else { xb1 };
                    let ym = p(xm);
                    if ym.abs() < yr.abs() {
                        xr = xm;
                    }
                }
                break;
            }
            yr = p(xr);
        }
    }
    xr
}

// #[inline]
// fn find_open(N: usize, p: impl Fn(f64) -> f64, dp: impl Fn(f64) -> f64, tol: f64) -> f64 {
//     let mut xr = 0.;
//     let yr = self.p.c[N];
//     if (yr < 0.) != (self.p.c[0] < 0.) {
//         find_open_max(N, p, dp, xr, yr, tol)
//     } else {
//         find_open_min(N, p, dp, xr, yr, tol)
//     }
// }

#[inline]
fn find_open_max(
    N: usize,
    p: impl Fn(f64) -> f64,
    dp: impl Fn(f64) -> f64,
    mut x0: f64,
    mut y0: f64,
    tol: f64,
) -> f64 {
    find_open_helper(N, p, dp, x0, y0, x0 + 1., tol, false)
}
#[inline]
fn find_open_min(
    N: usize,
    p: impl Fn(f64) -> f64,
    dp: impl Fn(f64) -> f64,
    mut x1: f64,
    mut y1: f64,
    tol: f64,
) -> f64 {
    find_open_helper(N, p, dp, x1, y1, x1 - 1., tol, true)
}

#[inline]
fn find_open_helper(
    N: usize,
    p: impl Fn(f64) -> f64,
    dp: impl Fn(f64) -> f64,
    mut xm: f64,
    mut ym: f64,
    mut xr: f64,
    tol: f64,
    openMin: bool,
) -> f64 {
    let mut delta = 1.;
    let mut yr = p(xr);

    let mut otherside: bool = (ym < 0.) != (yr < 0.);

    'main_loop: while yr != 0. {
        if otherside {
            if openMin {
                return find_closed(N, p, dp, xr, xm, yr, ym, tol);
            } else {
                return find_closed(N, p, dp, xm, xr, ym, yr, tol);
            }
        } else {
            'open_interval: loop {
                xm = xr;
                ym = yr;
                let dy = dp(xr);
                let dx = yr / dy;
                let xn = xr - dx;
                let dif = if openMin { xr - xn } else { xn - xr }; // Consider using |xr-xn|...
                if dif <= 0. && xn.is_finite() {
                    xr = xn;
                    if dif <= tol {
                        if xr == xm {
                            break 'main_loop;
                        };
                        let xs = xn - tol * if openMin { -1. } else { 1. };
                        let ys = p(xs);
                        let s = (ym < 0.) != (ys < 0.);
                        if s {
                            break 'main_loop;
                        };
                        xr = xs;
                        yr = ys;
                        continue 'open_interval;
                    }
                } else {
                    xr = if openMin { xr - delta } else { xr + delta };
                    delta += 2.;
                }
                yr = p(xr);
                otherside = (ym < 0.) != (yr < 0.);
                continue 'main_loop;
            }
        }
    }

    xr
}

#[inline]
pub fn roots_quadratic(p: &Polynomial<f64, 3>) -> [f64; 2] {
    let mut output = [f64::NAN; 2];
    let a = p.c[0];
    let b = p.c[1];
    let c = p.c[2];
    let delta = b * b - 4. * a * c;
    if delta > 0. {
        // Two real roots
        let d = delta.sqrt();
        let q = -0.5 * (b + d.copysign(b.signum()));
        let rv0 = q / a;
        let rv1 = c / q;
        output[0] = rv0.min(rv1);
        output[1] = rv0.max(rv1);
        return output;
    } else if delta < 0. { // Roots are complex conjugate pair, return NaNs
    } else {
        // One real root
        output[0] = -0.5 * b / a;
    }

    output
}

#[inline]
pub fn roots_cubic(f: &Polynomial<f64, 4>, tol: f64) -> [f64; 3] {
    let mut output = [f64::NAN; 3];
    let a = f.c[0] * 3.;
    let b_2 = f.c[1];
    let c = f.c[2];

    let df = Polynomial::<f64, 3>::from([a, 2. * b_2, c]);
    let p = { |x| f.eval(x) };
    let dp = { |x| df.eval(x) };

    let delta_4 = b_2 * b_2 - a * c;

    if delta_4 > 0. {
        let d_2 = delta_4.sqrt();
        let q = -(b_2 + (d_2 * if b_2.signum() < 0. { -1. } else { 1. }));
        let rv0 = q / a;
        let rv1 = c / q;
        let xa = rv0.min(rv1);
        let xb = rv0.max(rv1);

        let ya = p(xa);
        let yb = p(xb);

        if a.signum() == ya.signum() {
            output[0] = find_open_min(3, p, dp, xa, ya, tol);
            if (ya < 0.) != (yb < 0.) {
                output[1] = find_closed(3, p, dp, xa, xb, 0., 0., tol);
                output[2] = find_open_max(3, p, dp, xb, yb, tol);
            }
        } else {
            output[0] = find_open_max(3, p, dp, xb, yb, tol);
        }
    } else {
        let x_inf = -b_2 / a;
        let y_inf = p(x_inf);
        if a.signum() != y_inf.signum() {
            output[0] = find_open_max(3, p, dp, x_inf, y_inf, tol);
        } else {
            output[0] = find_open_min(3, p, dp, x_inf, y_inf, tol);
        }
    }

    output
}

#[inline]
pub fn roots_quartic(f: &Polynomial<f64, 5>, tol: f64) -> [f64; 4] {
    const N: usize = 4;
    let mut output = [f64::NAN; N];
    let df = Polynomial::<f64, 4>::from([4. * f.c[0], 3. * f.c[1], 2. * f.c[2], f.c[3]]);
    let derivRoots = roots_cubic(&df, tol);

    let p = { |x| f.eval(x) };
    let dp = { |x| df.eval(x) };
    let nd = derivRoots
        .iter()
        .map(|x| if x.is_finite() { 1 } else { 0 })
        .sum::<usize>();
    if ((N & 1) != 0) || ((N & 1) == 0 && nd > 0) {
        let mut nr = 0;
        let mut xa = derivRoots[0];
        let mut ya = p(xa);
        if ((ya < 0.) != (f.c[0] < 0.)) != ((N & 1) != 0) {
            output[0] = find_open_min(N, p, dp, xa, ya, tol);
            nr = 1;
        }
        for i in 1..nd {
            let xb = derivRoots[i];
            let yb = p(xb);
            if (ya < 0.) != (yb < 0.) {
                output[nr] = find_closed(N, p, dp, xa, xb, ya, yb, tol);
                nr += 1;
            }
            xa = xb;
            ya = yb;
        }
        if (ya < 0.) != (f.c[0] < 0.) {
            output[nr] = find_open_max(N, p, dp, xa, ya, tol);
            // nr += 1;
        }
    } else {
        if (N & 1) != 0 {
            // output[0] = find_open(N, p, dp, tol);
            todo!()
        }
    }

    output
}
