function misner_4d_to_polar(misner_T, misner_phi, y, z)
{
	var t = misner_T * CMath.exp(misner_phi/2) - CMath.exp(-misner_phi/2);
	var x = misner_T * CMath.exp(misner_phi/2) + CMath.exp(-misner_phi/2);
	
    var r = CMath.sqrt(x * x + y * y + z * z);
    var theta = CMath.atan2(CMath.sqrt(x * x + y * y), z);
    var phi = CMath.atan2(y, x);

    return [t, r, theta, phi];
}

misner_4d_to_polar
