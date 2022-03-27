function double_schwarzschild(t, p, phi, z)
{
    ///this functions performance would indicate that something very suboptimal is happening
    ///probably around the poles
	
	$cfg.M1.$default = 1;
	$cfg.M2.$default = 0.1;
	$cfg.z.$default = 2;
	
    var M1 = $cfg.M1;
    var M2 = $cfg.M2;

    var e = M2 - M1;
    var M = M1 + M2;

    var z0 = $cfg.z;

    var a1 = -0.5 * (M - e) - z0;
    var a2 = 0.5 * (M - e) - z0;
    var a3 = -0.5 * (M + e) + z0;
    var a4 = 0.5 * (M + e) + z0;

    ///idx [1, 4]
    function ak(idx)
    {
        if(idx == 1)
            return a1;

        if(idx == 2)
            return a2;

        if(idx == 3)
            return a3;

        if(idx == 4)
            return a4;
    }

    function Rk(idx)
    {
        return CMath.sqrt(p*p + (z - ak(idx)) * (z - ak(idx)));
    };

    function Yk(idx)
    {
        return Rk(idx) + ak(idx) - z;
    };

    function Yij(i1, j1)
    {
        return Rk(i1) * Rk(j1) + (z - ak(i1)) * (z - ak(j1)) + p*p;
    };

    var e2k = (Yij(4, 3) * Yij(2, 1) * Yij(4, 1) * Yij(3, 2)) / (4 * Yij(4, 2) * Yij(3, 1) * Rk(1) * Rk(2) * Rk(3) * Rk(4));

    var e_2U = (Yk(1) * Yk(3)) / (Yk(2) * Yk(4));
    var e_m2U = (Yk(2) * Yk(4)) / (Yk(1) * Yk(3));

    var dt = -e_2U;
    var dp = e_m2U * e2k;
    var dphi = e_m2U * p * p;
    var dz = e_m2U * e2k;

    var ret = [];
    ret.length = 16;

    ret[0 * 4 + 0] = dt;
    ret[1 * 4 + 1] = dp;
    ret[2 * 4 + 2] = dphi;
    ret[3 * 4 + 3] = dz;

    return ret;
}

double_schwarzschild
