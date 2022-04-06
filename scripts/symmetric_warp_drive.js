///https://arxiv.org/pdf/2010.11031.pdf
function symmetric_warp_drive(t, r, theta, phi)
{
    theta = M_PI/2;

    var rg = 1;
    var rk = rg;

    var a20 = (1 - rg / r);

    var a0 = CMath.sqrt(a20);

    var a1 = t / theta;

    var a2 = a20 + a1;

    var yrr0 = 1 / (1 - (rg / r));
    var ythetatheta0 = r * r;
    var yphiphi0 = r * r * CMath.sin(theta) * CMath.sin(theta);

    var gamma_0 = CMath.pow(r, 4) * CMath.sin(theta) * CMath.sin(theta) / (1 - rg/r);

    var littlea = rk * theta * CMath.pow(a0, -1);
    var littleb = rk * theta - CMath.sqrt(gamma_0);

	///this is only correct for radial geodesics unfortunately
    var Urt = (littlea * CMath.pow(a20 + t/theta, 3/2) - littleb) / (littlea * a0*a0*a0 - littleb);

    var yrr = Urt * yrr0;
    var ythetatheta = Urt * ythetatheta0;
    var yphiphi = Urt * yphiphi0;

    var dt = -a2;

    return [dt, yrr, ythetatheta, yphiphi];
}

symmetric_warp_drive
