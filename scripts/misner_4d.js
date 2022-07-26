function misner_4d(misner_T, misner_phi, y, z)
{
	var met = [];
	met.length = 16;

	///https://arxiv.org/pdf/1102.0907.pdf (25)
	var dT_dphi = -2;
	var dphi = -misner_T;
	var dy = 1;
	var dZ = 1;
	
	met[0 * 4 + 1] = 0.5 * dT_dphi;
	met[1 * 4 + 0] = 0.5 * dT_dphi;
	met[1 * 4 + 1] = dphi;
	met[2 * 4 + 2] = dy;
	met[3 * 4 + 3] = dZ;
	
    return met;
}

misner_4d
