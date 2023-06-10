use std::num::FpCategory;

/// Returns a specified-width string representation of the provided f64.
/// The absolute minimum width is 3, but this may panic with overflow for widths
/// under 7.
#[allow(dead_code)]
pub fn format_f64(n: f64, width: usize) -> String {
    assert!(width >= 3);
    let mut output = String::new();
    let width_used: usize;

    let class = n.classify();

    if let FpCategory::Normal | FpCategory::Subnormal = class {
        // TODO: Better handling of subnormal case

        let mut formatting_width = 2;

        let n_abs = n.abs();

        if n_abs > n {
            output.push('-');
            formatting_width += 1;
        }

        let has_exponent: bool;
        let exp_signed: isize = n_abs.abs().log10() as isize;
        if exp_signed == 0 {
            has_exponent = false;
        } else {
            has_exponent = true;
            formatting_width += 1;
            if exp_signed < 0 {
                formatting_width += 1; // Exponent is negative, we'll need a '-'
            }
            let exp_wdh = 1 + (exp_signed as f64).abs().log10() as usize;
            formatting_width += exp_wdh; // TODO: use int_log when stable
        }

        // TODO: Check trailing zeros and compact when possible
        if !has_exponent {
            output.push_str(&format!("{:.w$}", n_abs, w = width - formatting_width));
        } else {
            assert!(formatting_width <= width);
            output.push_str(&format!("{:1.w$e}", n_abs, w = width - formatting_width));
        }
        width_used = width;
    } else if let FpCategory::Zero = class {
        // No distinction is made for negative zero
        output.push_str("0.");
        width_used = 2;
    } else if let FpCategory::Infinite = class {
        if n == f64::INFINITY {
            output.push('∞');
            width_used = 1;
        } else {
            output.push_str("-∞");
            width_used = 2;
        }
    } else {
        output.push_str("NaN");
        width_used = 3;
    }

    let padding = width - width_used;
    for _ in 0..padding {
        output.push(' ');
    }
    output
}
