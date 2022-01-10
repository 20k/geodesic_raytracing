function cartesian_to_polar(t, x, y, z)
{
    var r = CMath.sqrt(x * x + y * y + z * z);
    var theta = CMath.atan2(CMath.sqrt(x * x + y * y), z);
    var phi = CMath.atan2(y, x);

    return [t, r, theta, phi];
}

cartesian_to_polar
