///https://www.sciencedirect.com/science/article/pii/S0370269319303375
function unequal_double_kerr(t, p, phi, z)
{
    var i = CMath.i;

    /*dual a1 = -0.09;
    dual a2 = 0.091;*/
	
	$cfg.m1.$default = 0.15;
	$cfg.m2.$default = 0.3;
	$cfg.fa1.$default = 1;
	$cfg.fa2.$default = -0.3;
	
	$cfg.R.$default = 4;

    var m1 = $cfg.m1;
    var m2 = $cfg.m2;

    var fa1 = $cfg.fa1;
    var fa2 = $cfg.fa2;

    var a1 = fa1 * m1;
    var a2 = fa2 * m2;
	
	var R = $cfg.R;
	
	/*var m1 = 0.15;
	var m2 = 0.3;
	var fa1 = 1;
	var fa2 = -0.3;
	var R = 4;
	
	var a1 = fa1 * m1;
	var a2 = fa2 * m2;*/

    /*var a1 = 0.3;
    var a2 = 0.1;

    var m1 = 0.4;
    var m2 = 0.4;
	
	var R = 4;*/
	
    var J = m1 * a1 + m2 * a2;
    var M = m1 + m2;
	
	$pin(J);
	$pin(M);

    ///https://www.wolframalpha.com/input/?i=%28%28a_1+%2B+a_2+-+x%29+*+%28R%5E2+-+M%5E2+%2B+x%5E2%29+%2F+%282+*+%28R+%2B+M%29%29%29+-+M+*+x+%2B+J+%3D+0+Solve+for+x
    ///https://www.wolframalpha.com/input/?i=%28%28k+-+a%29+*+%28B+%2B+a%5E2%29+%2F+C%29+-+M+*+a+%2B+J+solve+for+a

    var a = 0;

    {
        var k = a1 + a2;
        var B = R*R - M*M;
        var C = 2 * (R + M);

        var inner_val = CMath.pow(CMath.sqrt(CMath.pow(18 * B * k + 27 * C * J - 9 * C * k * M + 2 * k*k*k, 2) + 4 * CMath.pow(3 * B + 3 * C * M - k*k, 3)) + 18 * B * k + 27 * C * J - 9 * C * k * M + 2 *k*k*k, 1/3);

        var third_root_2 = CMath.pow(2, 1/3);

        a = (1 / (3 * third_root_2)) * inner_val - ((third_root_2 * (3 * B + 3 * C * M - k*k)) / (3 * inner_val)) + k/3;
    }
	
	$pin(a);

    var d1 = ((m1 * (a1 - a2 + a) + R * a) * (CMath.pow(R + M, 2) + a * a) + m2 * a1 * a*a) / CMath.pow(CMath.pow(R + M, 2) + a*a, 2);
    var d2 = ((m2 * (a2 - a1 + a) + R * a) * (CMath.pow(R + M, 2) + a * a) + m1 * a2 * a*a) / CMath.pow(CMath.pow(R + M, 2) + a*a, 2);

	$pin(d1);
	$pin(d2);

    var s1 = CMath.csqrt(m1 * m1 - a1 * a1 + 4 * m2 * a1 * d1);
    var s2 = CMath.csqrt(m2 * m2 - a2 * a2 + 4 * m1 * a2 * d2);
	
	$pin(s1);
	$pin(s2);

    ///R+ with a squiggle on
    var Rsp = CMath.psqrt(p * p + CMath.pow(z + 0.5 * R + s2, 2));
    var Rsn = CMath.psqrt(p * p + CMath.pow(z + 0.5 * R - s2, 2));

    var rsp = CMath.psqrt(p * p + CMath.pow(z - 0.5 * R + s1, 2));
    var rsn = CMath.psqrt(p * p + CMath.pow(z - 0.5 * R - s1, 2));

    var mu0 = (R + M - i * a) / (R + M + i * a);

	$pin(mu0);
	$pin(Rsp);
	$pin(Rsn);
	$pin(rsp);
	$pin(rsn);

    var rp = (1/mu0) *  (((s1 - m1 - i * a1) * (CMath.pow(R + M, 2) + a*a) + 2 * a1 * (m1 * a + i * M * (R + M))) /
                                  ((s1 - m1 + i * a1) * (CMath.pow(R + M, 2) + a*a) + 2 * a1 * (m1 * a - i * M * (R + M))))
                                  * rsp;

    var rn = (1/mu0) * (((-s1 - m1 - i * a1) * (CMath.pow(R + M, 2) + a*a) + 2 * a1 * (m1 * a + i * M * (R + M))) /
                                 ((-s1 - m1 + i * a1) * (CMath.pow(R + M, 2) + a*a) + 2 * a1 * (m1 * a - i * M * (R + M))))
                                 * rsn;

    var Rp = -mu0    *  (((s2 + m2 - i * a2) * (CMath.pow(R + M, 2) + a*a) - 2 * a2 * (m2 * a - i * M * (R + M))) /
                                  ((s2 + m2 + i * a2) * (CMath.pow(R + M, 2) + a*a) - 2 * a2 * (m2 * a + i * M * (R + M))))
                                 * Rsp;

    var Rn = -mu0    * (((-s2 + m2 - i * a2) * (CMath.pow(R + M, 2) + a*a) - 2 * a2 * (m2 * a - i * M * (R + M))) /
                                 ((-s2 + m2 + i * a2) * (CMath.pow(R + M, 2) + a*a) - 2 * a2 * (m2 * a + i * M * (R + M))))
                                 * Rsn;

	$pin(rp);
	$pin(rn);
	$pin(Rp);
	$pin(Rn);

    /*var A = R * R * (Rp - Rn) * (rp - rn) - 4 * sigma_sq * (Rp - rp) * (Rn - rn);
    var B = 2 * R * sigmap * ((R + 2 * sigmap) * (Rn - rp) - (R - 2 * sigmap) * (Rp - rn));*/

    var A = (R*R - CMath.pow(s1 + s2, 2)) * (Rp - Rn) * (rp - rn) - 4 * s1 * s2 * (Rp - rn) * (Rn - rp);
    var B = 2 * s1 * (R*R - s1*s1 + s2*s2) * (Rn - Rp) + 2 * s2 * (R * R + s1 * s1 - s2 * s2) * (rn - rp) + 4 * R * s1 * s2 * (Rp + Rn - rp - rn);

	$pin(A);
	$pin(B);

    var G = -z * B + s1 * (R * R - s1 * s1 + s2 * s2) * (Rn - Rp) * (rp + rn + R) + s2 * (R * R + s1 * s1 - s2*s2) * (rn - rp) * (Rp + Rn - R)
                     -2 * s1 * s2 * (2 * R * (rp * rn - Rp * Rn - s1 * (rn - rp) + s2 * (Rn - Rp)) + (s1 * s1 - s2 * s2) * (rp + rn - Rp - Rn));

	$pin(G);

    var K0 = ((CMath.pow(R + M, 2) + a*a) * (R*R - CMath.pow(m1 - m2, 2) + a*a) - 4 * m1*m1 * m2*m2 * a*a) / (m1 * m2 * (CMath.pow(R + M, 2) + a*a));

	$pin(K0);

    var w = 2 * a - (2 * CMath.Imaginary(G * (CMath.conjugate(A) + CMath.conjugate(B))) / (CMath.self_conjugate_multiply(A) - CMath.self_conjugate_multiply(B)));

	$pin(w);

    var f = (CMath.self_conjugate_multiply(A) - CMath.self_conjugate_multiply(B)) / CMath.Real((A + B) * (CMath.conjugate(A) + CMath.conjugate(B)));
    
	$pin(f);
	
	var e2g = (CMath.self_conjugate_multiply(A) - CMath.self_conjugate_multiply(B)) / CMath.Real(16 * CMath.pow(CMath.fabs(s1), 2) * CMath.pow(CMath.fabs(s2), 2) * K0*K0 * Rsp * Rsn * rsp * rsn);

	$pin(e2g);

    var dphi2 = w * w * -f;
    var dphi1 = (1/f) * p * p;
	
	$pin(dphi1);
	$pin(dphi2);

    var dt_dphi = f * w * 2;

    var dp = (1/f) * e2g;
    var dz = (1/f) * e2g;

    var ret = [];
    ret.length = 16;

    ret[0 * 4 + 0] = -f;
    ret[2 * 4 + 2] = dphi1 + dphi2;
    ret[0 * 4 + 2] = dt_dphi * 0.5;
    ret[2 * 4 + 0] = dt_dphi * 0.5;

    ret[1 * 4 + 1] = dp;
    ret[3 * 4 + 3] = dz;

    return ret;
}

unequal_double_kerr
