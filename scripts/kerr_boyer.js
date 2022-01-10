function kerr_metric(t, r, theta, phi)
{
    var rs = 1;

    var a = -0.5;
    var E = r * r + a * a * CMath.cos(theta) * CMath.cos(theta);
    var D = r * r  - rs * r + a * a;

    var c = 1;

    var ret = [];
    ret.length = 16;

    ret[0] = -(1 - rs * r / E) * c * c;
    ret[1 * 4 + 1] = E / D;
    ret[2 * 4 + 2] = E;
    ret[3 * 4 + 3] = (r * r + a * a + (rs * r * a * a / E) * CMath.sin(theta) * CMath.sin(theta)) * CMath.sin(theta) * CMath.sin(theta);
    ret[0 * 4 + 3] = 0.5 * -2 * rs * r * a * CMath.sin(theta) * CMath.sin(theta) * c / E;
    ret[3 * 4 + 0] = ret[0 * 4 + 3];

    return ret;
}

kerr_metric
