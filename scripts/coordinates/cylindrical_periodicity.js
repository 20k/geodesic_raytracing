function func(t, r, phi, z)
{
	function positive_fmod(a, b)
	{	
		var r = CMath.smooth_fmod(a, b);

		var rlt = CMath.lt(r, 0);

		return CMath.select(rlt, r + b, r);
	}
        
	var ret = [t, r, phi, z];
	
	ret[2] = positive_fmod(ret[2], 2*Math.PI);
	
	return ret;
}

func