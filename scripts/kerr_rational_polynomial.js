function kerr_rational_polynomial(t, r, X, phi)
{
    var m = 0.5;
    var a = -2;

    var dt = -(1 - 2 * m * r / (r * r + a * a * X * X));
    var dphidt = - (4 * a * m * r * (1 - X * X))/(r * r + a * a * X * X);
    var dr = (r * r + a * a * X * X) / (r * r - 2 * m * r + a * a);
    var dX = (r * r + a * a * X * X) / (1 - X * X);
    var dphi = (1 - X * X) * (r * r + a * a + (2 * m * a * a * r * (1 - X * X)) / (r * r + a * a * X * X));

    var ret = [];
    ret.length = 16;

    ret[0 * 4 + 0] = dt;
    ret[1 * 4 + 1] = dr;
    ret[2 * 4 + 2] = dX;
    ret[3 * 4 + 3] = dphi;
    ret[0 * 4 + 3] = dphidt * 0.5;
    ret[3 * 4 + 0] = dphidt * 0.5;

    return ret;
}

kerr_rational_polynomial
