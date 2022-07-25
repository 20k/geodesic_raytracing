function polar_to_misner_4d(t, r, theta, phi)
{
    var x = r * CMath.sin(theta) * CMath.cos(phi);
    var y = r * CMath.sin(theta) * CMath.sin(phi);
    var z = r * CMath.cos(theta);

	///https://arxiv.org/pdf/1102.0907.pdf 8 + 9
	var misner_phi = -2 * CMath.log((x - t)/2);
	
	var misner_T = (x*x - t*t) / 4;

    return [misner_T, misner_phi, y, z];
}

polar_to_misner_4d
